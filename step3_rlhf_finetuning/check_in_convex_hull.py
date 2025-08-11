import torch
import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm


def is_in_convex_hull(u, X):
    """
    Check if vector u lies in the convex hull of rows of X.
    """
    M, d = X.shape
    A_eq = np.vstack([X.T, np.ones(M)])
    b_eq = np.append(u, 1)
    c = np.zeros(M)
    bounds = [(0, None)] * M
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.success


def test_all_unsats_in_hull(unsat_path, train_path):
    # Load embeddings
    unsat_data = torch.load(unsat_path)
    train_data = torch.load(train_path)

    unsat = torch.stack([x["embedding"] for x in unsat_data]).cpu().numpy()
    train = torch.stack([x["embedding"] for x in train_data]).cpu().numpy()

    results = []
    for i in tqdm(range(unsat.shape[0]), desc="Testing convex hull inclusion"):
        u = unsat[i]
        inside = is_in_convex_hull(u, train)
        results.append(inside)

    return results


if __name__ == "__main__":
    unsat_path = "/efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_unsatisfactory_embeddings.pt"
    train_path = "/efs/shicheng/remax/step3_rlhf_finetuning/opt-1.3b_full-hh-rlhf_training_embeddings.pt"
    results = test_all_unsats_in_hull(unsat_path, train_path)
    
    print(f"Total unsatisfactory embeddings: {len(results)}")
    print(f"Inside convex hull: {sum(results)}")
    print(f"Outside convex hull: {len(results) - sum(results)}")
