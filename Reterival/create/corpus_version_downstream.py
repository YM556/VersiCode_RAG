import json
import re
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir",help="The target file directory",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/")
parser.add_argument("--prefix",default="downstream_application_code",)
parser.add_argument("--output_dir",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/downstream_application_code/version_corpus")
args = parser.parse_args()

library_list = []

def clear_source_code(item):
    version_pattern = re.compile(r'<version>(.*?)</version>')
    in_version_pattern = r'([^,]*)(?=,|$)'
    pattern = r'([\w\-\.]+)\s*(>=|<=|~=|==|>|<)\s*([\w\.\+\-]+)'
    filePath_pattern = re.compile(r'<filePath>(.*?)</filePath>')
    cleaned_text = re.sub(r'<version>(.*?)</version>|<filePath>(.*?)</filePath>', '', item['data'])
    versions = ''.join(version_pattern.findall(item["data"]))
    versions_list = re.findall(in_version_pattern,versions)
    filePath = ''.join(filePath_pattern.findall(item['data']))
    library_name = []
    version_num = []

    for dep in versions_list:
        try:
            match = re.match(pattern,dep)
            library_name.append(match.group(1))
            version_num.append(match.group(3))
        except Exception as e:
            print(dep)
            print(e)

    source_code = cleaned_text
    return library_name,version_num,filePath,source_code

def main():
    for root,dir,file in os.walk(args.dir):
        for filename in file:
            if filename == "downstream_application_code_raw_data.jsonl":
                print("filename:{}".format(filename))
                try:
                    with open(os.path.join(root,filename),"r") as f:
                        for line in tqdm(f,desc="Processing"):
                            item = json.loads(line.strip())
                            library_names,version_nums,file_path, source_code = clear_source_code(item)
                            for library_name,version_num in zip(library_names,version_nums):
                                item_dict = {"library_name":library_name,"version_num":version_num,"file_path":file_path,"source_code":source_code}
                                output_library_dir = os.path.join(args.output_dir,library_name)
                                if not os.path.exists(output_library_dir):
                                    os.mkdir(output_library_dir)
                                output_file = os.path.join(output_library_dir, version_num+".jsonl")
                                if os.path.exists(output_file):
                                    with open(output_file,'a') as jsonl_file:
                                        jsonl_file.write(json.dumps(item_dict) + '\n')
                                else:
                                    with open(output_file,'w') as jsonl_file:
                                        jsonl_file.write(json.dumps(item_dict) + '\n')

                except Exception as e:
                    print(f"File error in {filename}: {e}")

if __name__ == '__main__':
    main()




