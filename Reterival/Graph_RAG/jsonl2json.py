import json

file_path = "/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/torch_old_to_new.jsonl"
output_file = "/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/torch_old_to_new.json"
data_list = []
with open(file_path,"r") as f:
    for idx,line in enumerate(f):
        item = json.loads(line)
        data_list.append(item)
lodic = {"data":data_list}
with open(output_file,"w") as of:
   json.dump(lodic,of,indent=4,ensure_ascii=False)