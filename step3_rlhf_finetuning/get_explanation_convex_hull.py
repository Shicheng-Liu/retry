import torch, json, argparse, os
import numpy as np
from tqdm import tqdm
import cvxpy as cp


def parse_args():
    parser = argparse.ArgumentParser(description="Project unsatisfactory embeddings onto convex hull of training embeddings")
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--unsatisfactory_embedding_path", type=str, required=True)
    parser.add_argument("--training_embedding_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_explanation_path", type=str, required=True,
                        help="Path to save matched training examples. Will also save weights to xxx_weights.json")
    return parser.parse_args()


def load_embeddings_with_index(path):
    data = torch.load(path)
    if isinstance(data, list) and isinstance(data[0], dict):
        embeddings = torch.stack([item["embedding"] for item in data])
        indices = [item["index"] for item in data]
        return embeddings, indices
    else:
        raise TypeError(f"Expected list of dicts with 'embedding' and 'index', got {type(data)}")


def project_to_convex_hull(point, hull_points):
    """
    Project `point` onto the convex hull of `hull_points`.
    Always returns a convex combination (weights).
    """
    X = hull_points  # shape: (k, d)
    k, d = X.shape
    w = cp.Variable(k)

    objective = cp.Minimize(cp.sum_squares(X.T @ w - point))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if w.value is not None:
        weights = w.value
        weights = weights / weights.sum()  # ensure numerical stability
        return weights
    else:
        raise ValueError("Convex projection failed")


def main():
    args = parse_args()
    N = args.num_neighbors
    BATCH_SIZE = args.batch_size

    unsatisfactory_embeddings, _ = load_embeddings_with_index(args.unsatisfactory_embedding_path)
    training_embeddings, training_indices = load_embeddings_with_index(args.training_embedding_path)

    all_matched_indices = set()
    explanation_data = []

    for i in tqdm(range(0, unsatisfactory_embeddings.size(0), BATCH_SIZE), desc="Projecting"):
        batch = unsatisfactory_embeddings[i:i + BATCH_SIZE]

        for b_idx, unsat in enumerate(batch):
            unsat_np = unsat.cpu().numpy()
            dists = torch.cdist(unsat.unsqueeze(0), training_embeddings, p=2).squeeze(0)
            sorted_indices = torch.argsort(dists)

            selected = sorted_indices[:N]
            candidates_np = training_embeddings[selected].cpu().numpy()
            weights = project_to_convex_hull(unsat_np, candidates_np)

            match_info = []
            for j, tensor_idx in enumerate(selected):
                original_idx = training_indices[tensor_idx.item()]
                all_matched_indices.add(original_idx)
                match_info.append({
                    "train_index": original_idx,
                    "weight": float(weights[j])
                })

            explanation_data.append({
                "unsat_index": i + b_idx,
                "matched_type": "projected_convex_hull",
                "matches": match_info
            })

    # Load full training data
    with open(args.train_data_path, "r", encoding="utf-8") as f:
        full_train_data = [json.loads(line) for line in f]

    # Save matched examples
    subset_data = []
    for i in sorted(all_matched_indices):
        item = full_train_data[i].copy()
        item["chosen"], item["rejected"] = item.get("rejected"), item.get("chosen")
        subset_data.append(item)

    # Save matched training examples
    with open(args.output_explanation_path, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)

    # Save projection weights
    base, ext = os.path.splitext(args.output_explanation_path)
    weight_path = base + "_weights" + ext
    with open(weight_path, "w", encoding="utf-8") as f:
        json.dump(explanation_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
