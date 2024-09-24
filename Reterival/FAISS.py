import os.path

import faiss
import numpy as np
import torch.cuda
from sentence_transformers import SentenceTransformer
import json
import argparse
from tqdm import tqdm



def compute_NDCG(relvant,k):
    idcg=0.0
    for i in range(1,k+1):
        idcg+=relvant

def main():
    global doc_id, score
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",default="BAAI/bge-large-en-v1.5",
                        help="Using sentence transformer to embedding ")
    parser.add_argument("--corpus", default="./datasets/downstream_application_code/corpus.jsonl",
                        help="File to your corpus")
    parser.add_argument("--queries", default="/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/library_source_code/queries_line.jsonl",
                        help="File to your queries")
    parser.add_argument("--output_dir",default="./output/library_source_code")
    parser.add_argument("--cache_dir",default="/root/.cache")
    parser.add_argument("--task", type=str,default="line",help="The query from different level. including block, line,token")
    parser.add_argument("--sample_num", type=int,default=200)




    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device is {}".format(device))



    corpus_file = args.corpus
    # query_file = args.queries

    from transformers import AutoTokenizer, AutoModel

    # model_ckpt = "avsolatorio/GIST-large-Embedding-v0"

    model_ckpt="avsolatorio/GIST-large-Embedding-v0"
    model = SentenceTransformer(model_ckpt)


    # 读取语料库
    with open(corpus_file, 'r') as f:
        corpus = [json.loads(line) for line in f]

    # if args.sample_num==0:
    #     sample_num = len(corpus)
    # else:
    #     sample_num = args.sample_num

    # 生成语料库嵌入向量
    corpus_embeddings = model.encode([doc['text'] for doc in tqdm(corpus[:100],desc="Encoding Corpus")])
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    print("Encoding finished")
    print(corpus_embeddings.shape)
    # 读取查询
    query_files = ["queries_line.jsonl","queries_block.jsonl"]

    query_file_prefix = "/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/downstream_application_code"

    for query_file_post in query_files:
        query_file = os.path.join(query_file_prefix,query_file_post)
        print("query_file:{}".format(query_file))
        with open(query_file, 'r') as f:
            queries = [json.loads(line) for line in tqdm(f,desc="Reading queries")]

        # import ipdb
        # ipdb.set_trace()

        print("Reading finished")
        # 生成查询嵌入向量
        query_embeddings = model.encode([query['text'] for query in queries],device=device)
        # 计算模长
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        # # 建立 FAISS 索引
        # dimension = corpus_embeddings.shape[1]
        # index = faiss.IndexFlatL2(dimension)

        # 计算余弦相似度
        dimension = corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(corpus_embeddings))

        # 将索引移动到 GPU
        # res = faiss.StandardGpuResources()  # 创建 GPU 资源
        # index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
        # # 添加语料库嵌入向量到索引
        # index_gpu.add(np.array(corpus_embeddings))
        index.add(np.array(corpus_embeddings))

        # 检索前10个相关文档
        k = 10
        distances, indices = index.search(np.array(query_embeddings), k)

        # 处理检索结果
        results = []
        ndcg_results = []
        for query_idx, query in tqdm(enumerate(queries),desc="Processing Results", total=len(queries)):
            query_id = query['id']
            idcg = 0.0
            dcg = 0.0
            for i in range(k):
                doc_id = corpus[indices[query_idx, i]]['id']
                score = distances[query_idx, i]
                idcg += score
                dcg += score/np.log2(i+2)
            results.append({
                'query_id': query_id,
                'doc_id': doc_id,
                'score': str(score)
            })
            ndcg = dcg/idcg
            ndcg_results.append({"query_id":query_id,"NDCG@10":ndcg})

        write_path_f = os.path.join("./output/tmp","faiss_res.jsonl")
        write_path_n = os.path.join("./output/tmp","NDCG@10_res.jsonl")
        # write_path_f = os.path.join(args.output_dir,query_file_post[8:-6],"faiss_res.jsonl")
        # write_path_n = os.path.join(args.output_dir,query_file_post[8:-6],"NDCG@10_res.jsonl")
        print("Write Path:{}".format(write_path_f))
        # 输出结果
        with open(write_path_f, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

        with open(write_path_n, 'w') as f:
            for result in ndcg_results:
                f.write(json.dumps(result) + '\n')



if __name__ =='__main__':
    main()
