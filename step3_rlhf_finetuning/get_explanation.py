import torch, json, argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--num_neighbors",
        type=int,
        help="number of neighbor training data to find",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        required=True,
    )
    parser.add_argument(
        "--unsatisfactory_embedding_path",
        type=str,
        help="unsatisfactory embedding path",
        required=True,
    )
    parser.add_argument(
        "--training_embedding_path",
        type=str,
        help="training embedding path",
        required=True,
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="train data path",
        required=True,
    )
    parser.add_argument(
        "--output_explanation_path",
        type=str,
        help="output explanation path",
        required=True,
    )

    args = parser.parse_args()

    return args

# Config

def load_embeddings_with_index(path):
    data = torch.load(path)
    if isinstance(data, list) and isinstance(data[0], dict):
        embeddings = torch.stack([item["embedding"] for item in data])  # shape: (N2, D)
        indices = [item["index"] for item in data]  # original indices
        return embeddings, indices
    else:
        raise TypeError(f"Expected list of dicts with 'embedding' and 'index', got {type(data)}")

def main():
    args = parse_args()
    N = args.num_neighbors  # Top-N nearest neighbors
    BATCH_SIZE = args.batch_size
    unsatisfactory_embedding_path = args.unsatisfactory_embedding_path
    training_embedding_path = args.training_embedding_path
    train_data_path = args.train_data_path
    output_explanation_path = args.output_explanation_path

    # Load both sets
    unsatisfactory_embeddings, _ = load_embeddings_with_index(unsatisfactory_embedding_path)         # unsatisfactory embeddings
    training_embeddings, training_indices = load_embeddings_with_index(training_embedding_path)  # training embeddings with original indices

    all_matched_indices = set()

    for i in tqdm(range(0, unsatisfactory_embeddings.size(0), BATCH_SIZE), desc="Finding explanation"):
        batch = unsatisfactory_embeddings[i:i + BATCH_SIZE]
        dist = torch.cdist(batch, training_embeddings, p=2)
        # print(dist)
        # print(torch.min(dist))
        # print(torch.max(dist))
        _, topk_tensor_indices = torch.topk(dist, k=N, dim=1, largest=False)

        for row in topk_tensor_indices:
            for tensor_idx in row:
                original_idx = training_indices[tensor_idx.item()]
                all_matched_indices.add(original_idx)

    # === Step 3: Load Full Training Set (JSONL) ===
    with open(train_data_path, "r", encoding="utf-8") as f:
        full_train_data = [json.loads(line) for line in f]

    # === Step 4: Extract Subset and Save to JSON ===
    subset_data = []

    for i in sorted(all_matched_indices):
        item = full_train_data[i].copy()
        item["chosen"], item["rejected"] = item.get("rejected"), item.get("chosen")
        subset_data.append(item)

    with open(output_explanation_path, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()