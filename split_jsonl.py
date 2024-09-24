import os
import json


def split_jsonl(file_path, num_parts=4):
    # 读取所有行到一个列表中
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 计算每部分的大小
    total_lines = len(lines)
    part_size = total_lines // num_parts

    # 分割并写入到多个文件中
    for i in range(num_parts):
        start_index = i * part_size
        # 最后一部分包括剩余的所有行
        if i == num_parts - 1:
            end_index = total_lines
        else:
            end_index = (i + 1) * part_size

        part_lines = lines[start_index:end_index]

        # 创建新的文件名
        part_file_path = os.path.join('/Volumes/kjz-SSD/Datasets/raw_data_docstring_and_repostory/raw_data_docstring',f'raw_data_docstring_part_{i + 1}.jsonl')

        # 将该部分的内容写入新的文件
        with open(part_file_path, 'w', encoding='utf-8') as part_file:
            part_file.writelines(part_lines)

        print(f'Successfully created {part_file_path}')


# 使用示例
split_jsonl('/Volumes/kjz-SSD/Datasets/raw_data_docstring_and_repostory/raw_data_docstring.jsonl')
