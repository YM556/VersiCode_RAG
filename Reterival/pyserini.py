import os.path
import numpy as np
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import JWhiteSpaceAnalyzer
import torch
import subprocess

def index_single_dataset(output_metadir, index_dir, dataset):
    input_dir = os.path.join(output_metadir, f"{dataset}_corpus")
    output_dir = os.path.join(index_dir, f"{dataset}_corpus")
    subprocess.run(["python", "-m", "pyserini.index",
                    "-collection", "JsonCollection",
                    "-generator", "DefaultLuceneDocumentGenerator",
                    "-threads", "20",
                    "-input", input_dir,
                    "-index", output_dir,
                    "-storePositions", "-storeDocvectors", "-storeContents", "-pretokenized"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                        help="Using sentence transformer to embedding ")
    parser.add_argument("--corpus", default="./datasets/corpus.jsonl",
                        help="File to your corpus")
    parser.add_argument("--queries", default="./datasets/queries_block.jsonl",
                        help="File to your queries")
    parser.add_argument("--output_dir", default="../output")
    parser.add_argument("--cache_dir", default="/root/.cache")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is {}".format(device))

    corpus_file = args.corpus
    query_file = args.queries

    # 加载模型
    model = SentenceTransformer(args.model)

    # 读取语料库
    with open(corpus_file, 'r') as f:
        corpus = [json.loads(line) for line in f]

    # 生成语料库嵌入向量
    corpus_embeddings = model.encode([doc['text'] for doc in tqdm(corpus, desc="Encoding Corpus")],device=device)

    # 创建 Pyserini 索引
    index_path = os.path.join(args.output_dir, "index")
    os.makedirs(index_path, exist_ok=True)
    with open(os.path.join(index_path, "corpus.jsonl"), 'w') as f:
        for doc in corpus:
            f.write(json.dumps({"id": doc['id'], "text": doc['text']}) + '\n')

    # 索引文档
    os.system(f'python -m pyserini.index -collection JsonCollection '
              f'-generator DefaultLuceneDocumentGenerator '
              f'-threads 20 -input {index_path} -index {index_path}/index '
              f'-storePositions -storeDocvectors -storeContents -pretokenized')

    # 创建搜索器
    searcher = SimpleSearcher(os.path.join(index_path, 'index'))

    # 读取查询
    with open(query_file, 'r') as f:
        queries = [json.loads(line) for line in tqdm(f, desc="Reading queries")]

    # 检索前5个相关文档
    results = []
    k = 5
    for query in tqdm(queries, desc="Searching"):
        query_id = query['id']
        hits = searcher.search(query['text'], k)

        for hit in hits:
            results.append({
                'query_id': query_id,
                'doc_id': hit.docid,
                'score': hit.score
            })

    # 输出结果
    with open(os.path.join(args.output_dir, "pyserini_res.jsonl"), 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
