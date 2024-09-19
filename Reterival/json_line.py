import json
import re

def process_line(line):
    # 在这里对每一行数据进行处理
    item = json.loads(line)
    print(item)
    # 你可以在这里添加更多操作
def clear_source_code(item):
    version_pattern = re.compile(r'<version>(.*?)</version>')
    pattern = r'([\w\.\-]+)\s*==\s*([\w\.\-]+)'
    filePath_pattern = re.compile(r'<filePath>(.*?)</filePath>')
    cleaned_text = re.sub(r'<version>(.*?)</version>|<filePath>(.*?)</filePath>', '', item['data'])
    version = ''.join(version_pattern.findall(item["data"]))
    filePath = ''.join(filePath_pattern.findall(item['data']))
    try:
        match = re.match(pattern,version)
        library_name = match.group(1)
        version_num =  match.group(2)
    except Exception as e:
        print(version)
        print(e)

    source_code = cleaned_text
    item_dict = {"library_name":library_name,"version_num":version_num,"file_path":filePath,"source_code":source_code}
    return item_dict






# 示例用法
filepath = "/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/library_source_code_raw_data1.jsonl"

# 定义目标条件的函数
def target_condition(item):
    return item["library_name"]=="streamlit"  # 假设我们想从 id 为 3 的行开始


import json
import re
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir",help="The target file directory",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code")
parser.add_argument("--prefix",default="library_source_code",)
parser.add_argument("--output_dir",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus/tmp")
args = parser.parse_args()

library_list = []

def main():
        try:
            target_line_num = 0
            with open("/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/library_source_code_raw_data1.jsonl","r") as f:
                # print("filename:{}".format(filename))
                for current_line_num,line in tqdm(enumerate(f),desc="Processing"):
                    item = json.loads(line.strip())
                    item_dict = clear_source_code(item)
                    if target_condition(item_dict):
                        target_line_num = current_line_num
                        print(target_line_num)
                    if current_line_num >= target_line_num and target_line_num !=0:
                        output_file = os.path.join(args.output_dir,item_dict["library_name"]+".jsonl")
                        if os.path.exists(output_file):
                            with open(output_file,'a') as jsonl_file:
                                json.dump(item_dict,jsonl_file)
                                jsonl_file.write('\n')
                        else:
                            with open(output_file,'w') as jsonl_file:
                                print(item_dict["library_name"])
                                json.dump(item_dict, jsonl_file)
                                jsonl_file.write('\n')
        except Exception as e:
            print("Error_occured")
            print(item_dict["library_name"])
            print(f"File error in : {e}")

if __name__ == '__main__':
    main()




