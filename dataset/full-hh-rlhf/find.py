import json

filename = "validation.jsonl"
fields_to_check = ["prompt", "response", "chosen", "rejected"]

with open(filename, "r", encoding="utf-8") as f:
    buffer = ""
    for i, line in enumerate(f, 1):
        buffer += line
        try:
            obj = json.loads(buffer)
            buffer = ""

            for field in fields_to_check:
                if field not in obj:
                    print(f"❌ Line {i}: missing field '{field}'")
                elif obj[field] is None:
                    print(f"❌ Line {i}: field '{field}' is null")

        except json.JSONDecodeError:
            continue
