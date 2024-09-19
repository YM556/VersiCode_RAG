import json
import jsonlines



def load_jsonlines(file):
    with open(file,'r') as f:
        json_data = json.load(f)
        json_list = json_data["data"]
    return json_list



def write_jsonl(data,file):
    with jsonlines.open(file,mode="w") as writer:
        writer.write_all(data)




