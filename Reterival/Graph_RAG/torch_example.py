import json

with open ("/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/code_editing_old_to_new.json","r") as input_file:
    lodic = json.load(input_file)
    data_list = lodic["data"]

with open("/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/torch_old_to_new.jsonl","w") as output_file:
    for item in data_list:
        if item["dependency"] == "torch":
            output_file.write(json.dumps(item)+'\n')