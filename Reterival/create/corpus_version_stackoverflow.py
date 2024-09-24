import json
import re
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir",help="The target file directory",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/")
parser.add_argument("--prefix",default="stackoverflow",)
parser.add_argument("--output_dir",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/stackoverflow/version_corpus")
args = parser.parse_args()

library_list = []

def clear_source_code(item):
    title = item["title"]
    tags_list = item["tags"]
    question_body = item["question_body"]
    answer_body = item["answer_body"]

    return tags_list,question_body+answer_body

def main():
    for root,dir,file in os.walk(args.dir):
        for filename in file:
            if filename=="stackoverflow_raw_data.jsonl":
                print("filename:{}".format(filename))
                try:
                    with open(os.path.join(root,filename),"r") as f:
                        for line in tqdm(f,desc="Processing"):
                            item = json.loads(line.strip())
                            library_names,question_answer = clear_source_code(item)
                            for library_name in library_names:
                                item_dict = {"library_name":library_name,"question_answer":question_answer}
                                output_file = os.path.join(args.output_dir,library_name+".jsonl")
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




