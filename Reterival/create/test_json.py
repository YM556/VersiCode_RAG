import json
import jsonlines

file_path = "/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/torch_new_to_old.jsonl"
with open(file_path, 'r') as f:
    lodic = json.load(f)
    data_list = lodic["data"]
    for idx,line in enumerate(data_list):
        print(idx)
        print(line)