import json
import os
import sys

raw_path = "/efs/shicheng/remax/dataset/tldr_raw"
out_path = "/efs/shicheng/remax/dataset/tldr"
os.makedirs(out_path, exist_ok=True)

splits = ["train", "test"]
for split in splits:
    data_path = os.path.join(raw_path, split + ".jsonl")
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    res = []
    for line in data:
        prompt = "SUBREDDIT: r/{0}\n".format(line["info"]["subreddit"]) \
               + "TITLE: {0}\n".format(line["info"]["title"]) \
               + "POST: {0}\n".format(line["info"]["post"]) \
               + "TL;DR:"

        summaries = [line["summaries"][0]["text"], line["summaries"][1]["text"]]
        choice = line["choice"]
        chosen = " " + summaries[choice].strip()
        rejected = " " + summaries[0 if choice == 1 else 1].strip()
        res.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if split == "test":
        val_data = res[:500]
        test_data = res[500:]

        # Save first 500 to validation.jsonl
        with open(os.path.join(out_path, "validation.jsonl"), "w") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Save remaining test to test.jsonl
        with open(os.path.join(out_path, "test.jsonl"), "w") as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        # Save full training data to train.jsonl
        with open(os.path.join(out_path, "train.jsonl"), "w") as f:
            for item in res:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
