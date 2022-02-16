import json
import os


for mode in ["slobench"]:
    tasks_list = ["boolq", "cb", "copa", "multirc", "rte", "wsc"]
    for task in tasks_list:
        if not os.path.isdir(f"{os.getcwd()}/tasks/configs/{mode}"):
            os.makedirs(f"{os.getcwd()}/tasks/configs/{mode}")
        json_file = {
            "task": f"{task}",
            "paths": {
                "train": f"{os.getcwd()}/tasks/data/{mode}/{task}/train.jsonl",
                "val": f"{os.getcwd()}/tasks/data/{mode}/{task}/val.jsonl",
                "test": f"{os.getcwd()}/tasks/data/{mode}/{task}/test.jsonl",
                "test_with_answers": f"{os.getcwd()}/tasks/data/{mode}/{task}/test_answered.jsonl",
            },
            "name": f"{task}"
        }
        with open(f"{os.getcwd()}/tasks//configs/{mode}/{task}_config.json", "w") as f:
            json.dump(json_file, f)
