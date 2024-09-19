import argparse
import json
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--sample_num",type=int,default=200)
parser.add_argument("--sample_source",type=str,help="choices downstream_application_code,library_source_code stackoverflow")
parser.add_argument("--loc",help="local remote",default="remote")
parser.add_argument("--output_dir",type=str,default="samples")
args = parser.parse_args()

if args.loc=="local":
    prefix_path = "/Volumes/kjz-SSD/Datasets"
else:
    prefix_path = '/root/autodl-tmp/'

json_file = os.path.join(prefix_path,"VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion",args.sample_source,args.sample_source+"_token.json")

with open(json_file,"r",encoding="utf-8") as f:
    lodict = json.load(f)
    _data_list = lodict["data"]
    if len(_data_list)<args.sample_num:
        raise "Too many samples"
import random
data_list_ = [data for data in _data_list if data["dependency"]!="numpy"]
data_list = random.sample(data_list_,args.sample_num)
data_dict = {"data":data_list}
save_json_path = os.path.join(prefix_path,"VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion",args.sample_source,args.output_dir,args.sample_source+"_token.json")
with open(save_json_path,"w+",encoding="utf-8") as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)
