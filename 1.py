import json

path = (
    "/home/kg798/grad_std_llm/data/step3_1-shot+knowledge/step3_1-shot+knowledge.jsonl"
)
with open(path, "r") as file:
    for line in file:
        data = json.loads(line)
        print(data)
        break
