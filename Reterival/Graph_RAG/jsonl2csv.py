import json
import csv

# 输入的 JSONL 文件名
jsonl_file = '/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/library_source_code/torch_samples.jsonl'
# 输出的 CSV 文件名
csv_file = '/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/library_source_code/torch_entity.csv'

# 读取 JSONL 文件
with open(jsonl_file, 'r', encoding='utf-8') as infile:
    # 打开 CSV 文件准备写入
    with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None

        # 逐行读取 JSONL 文件
        for line in infile:
            # 将每行转换为 JSON 对象
            data = json.loads(line)

            # 如果 writer 还没有初始化，则初始化
            if writer is None:
                # 使用 JSON 对象的键作为 CSV 的列名
                writer = csv.DictWriter(outfile, fieldnames=data.keys())
                writer.writeheader()  # 写入表头

            # 写入 JSON 对象的值作为 CSV 的一行
            writer.writerow(data)
