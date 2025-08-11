import torch, json, argparse
from tqdm import tqdm
from scipy.optimize import linprog
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Find convex explanations for unsatisfactory embeddings")
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--unsatisfactory_embedding_path", type=str, required=True)
    parser.add_argument("--training_embedding_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_explanation_path", type=str, required=True,
                        help="Path to save matched training examples. Will also save weights to this_path + 'weights'")
    return parser.parse_args()


def load_embeddings_with_index(path):
    data = torch.load(path)
    if isinstance(data, list) and isinstance(data[0], dict):
        embeddings = torch.stack([item["embedding"] for item in data])
        indices = [item["index"] for item in data]
        return embeddings, indices
    else:
        raise TypeError(f"Expected list of dicts with 'embedding' and 'index', got {type(data)}")


def convex_combination_weights(point, hull_points):
    n = hull_points.shape[0]
    c = np.zeros(n)
    A_eq = np.vstack([hull_points.T, np.ones(n)])
    b_eq = np.append(point, 1)
    bounds = [(0, None)] * n
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        return True, res.x
    else:
        return False, None


def main():
    args = parse_args()
    N = args.num_neighbors
    BATCH_SIZE = args.batch_size

    unsatisfactory_embeddings, _ = load_embeddings_with_index(args.unsatisfactory_embedding_path)
    training_embeddings, training_indices = load_embeddings_with_index(args.training_embedding_path)

    all_matched_indices = set()
    explanation_data = []

    for i in tqdm(range(0, unsatisfactory_embeddings.size(0), BATCH_SIZE), desc="Finding explanation"):
        batch = unsatisfactory_embeddings[i:i + BATCH_SIZE]

        for b_idx, unsat in enumerate(batch):
            unsat_np = unsat.cpu().numpy()
            dists = torch.cdist(unsat.unsqueeze(0), training_embeddings, p=2).squeeze(0)
            sorted_indices = torch.argsort(dists)

            found = False
            for k in range(3, 100):
                selected = sorted_indices[:k]
                candidates_np = training_embeddings[selected].cpu().numpy()
                in_hull, weights = convex_combination_weights(unsat_np, candidates_np)
                if in_hull:
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
                        "matched_type": "convex_hull",
                        "matches": match_info
                    })
                    found = True
                    break

            if not found:
                top1_idx = sorted_indices[0].item()
                original_idx = training_indices[top1_idx]
                all_matched_indices.add(original_idx)
                explanation_data.append({
                    "unsat_index": i + b_idx,
                    "matched_type": "nearest_neighbor",
                    "matches": [{"train_index": original_idx, "weight": 1.0}]
                })

    # Load full train data
    with open(args.train_data_path, "r", encoding="utf-8") as f:
        full_train_data = [json.loads(line) for line in f]

    # Extract matched training examples (prompt/chosen/rejected)
    subset_data = []
    for i in sorted(all_matched_indices):
        item = full_train_data[i].copy()
        item["chosen"], item["rejected"] = item.get("rejected"), item.get("chosen")
        subset_data.append(item)

    # === Save training examples to output_explanation_path ===
    with open(args.output_explanation_path, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)

    # === Save convex weights and matched indices to xxx_weights.json ===
    import os
    base, ext = os.path.splitext(args.output_explanation_path)
    weight_path = base + "_weights" + ext
    with open(weight_path, "w", encoding="utf-8") as f:
        json.dump(explanation_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
