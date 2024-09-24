import json
import re
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir",help="The target file directory",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code")
parser.add_argument("--prefix",default="library_source_code",)
parser.add_argument("--output_dir",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus")
args = parser.parse_args()

library_list = []

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

def main():
    for root,dir,file in os.walk(args.dir):
        for filename in file:
            if filename=="library_source_code_raw_data2.jsonl":
                print("filename:{}".format(filename))
                try:
                    with open(os.path.join(root,filename),"r") as f:
                        # print("filename:{}".format(filename))
                        for line in tqdm(f,desc="Processing"):
                            item = json.loads(line.strip())
                            item_dict = clear_source_code(item)
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
                    print(f"File error in {filename}: {e}")

if __name__ == '__main__':
    main()




