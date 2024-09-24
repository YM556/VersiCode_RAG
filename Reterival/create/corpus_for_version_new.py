import json
import re
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dir",help="The target file directory",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus/new_tmp")
parser.add_argument("--output_dir",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus/new_tmp")
parser.add_argument("--postfix",default=".jsonl")
args = parser.parse_args()

library_list = []
def main():
    for filename in os.listdir(args.dir):
        if filename.endswith('.jsonl'):
            # 读取 JSONL 文件并处理数据
            jsonl_file_path = os.path.join(args.dir, filename)
            dirpath = os.path.join(args.dir, filename[:-6])
            os.mkdir(dirpath)
            with open(jsonl_file_path, 'r', encoding='utf-8') as fr:
                data_dict = {}
                for line in fr:
                    item = json.loads(line.strip())
                    # 假设您要根据字典中的某个特定字段进行分组
                    _version = item.get('version_num')  # 替换为您实际需要的键
                    if _version:
                        if _version not in data_dict:
                            data_dict[_version] = []
                        data_dict[_version].append(item)

            # 为每个特定字段值创建文件夹并写入数据
            for version, items in data_dict.items():
                output_subfolder = dirpath
                output_file_path = os.path.join(output_subfolder,version+".jsonl")
                with open(output_file_path, 'w', encoding='utf-8') as fw:
                    for item in items:
                        fw.write(json.dumps(item) + '\n')  # 将每个字典写入文件

if __name__ == '__main__':
    main()




