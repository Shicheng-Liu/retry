import json

input_path = "test.jsonl"  # replace with your actual path
val_output_path = "validation.json"
test_output_path = "test.json"

val_data = []
test_data = []

with open(input_path, "r", encoding="utf-8") as infile:
    for i, line in enumerate(infile):
        obj = json.loads(line)
        if i < 500:
            val_data.append(obj)
        else:
            test_data.append(obj)

# Save validation split
with open(val_output_path, "w", encoding="utf-8") as val_file:
    json.dump(val_data, val_file)

# Save test split
with open(test_output_path, "w", encoding="utf-8") as test_file:
    json.dump(test_data, test_file)