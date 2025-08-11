import json

input_path = "dataset/full-hh-rlhf/test.json"
val_output_path = "dataset/full-hh-rlhf/validation.json"
test_output_path = "dataset/full-hh-rlhf/test.json"

with open(input_path, "r") as infile:
    lines = infile.readlines()

# First 500 lines go to validation, rest to test
val_lines = lines[:500]
test_lines = lines[500:]

with open(val_output_path, "w") as val_file:
    for line in val_lines:
        val_file.write(line)

with open(test_output_path, "w") as test_file:
    for line in test_lines:
        test_file.write(line)

print(f"Validation set: {len(val_lines)} entries")
print(f"Test set: {len(test_lines)} entries")
