import json
from pyserini.search.faiss import FaissSearcher,TctColBertQueryEncoder
from tqdm import tqdm
import torch
import sys

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')

# 读取语料库并编码
corpus_embeddings = {}
with open('/root/autodl-tmp/VersiCode-RAG/Reterival/pyserini_retrieve/corpus/edit.jsonl', 'r') as f:
    for line in tqdm(f, desc="Encoding corpus", file=sys.stdout):
        doc = json.loads(line)
        corpus_embeddings[doc['id']] = encoder.encode(doc['contents']).to(device)

# 构建索引
index_dir = '/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/library_source_code/dense_index'
searcher = FaissSearcher.from_prebuilt_index(
    'msmarco-passage-tct_colbert-hnsw',
    encoder
)

# # if not searcher.index_exists():
# #     index_writer = IndexCollection(SimpleFSDirectory(index_dir))
# #     index_writer.init_index(len(corpus_embeddings))
# #
# #     for doc_id, embedding in tqdm(corpus_embeddings.items(), desc="Indexing documents", file=sys.stdout):
# #         index_writer.add_document(doc_id, embedding.cpu().numpy())
#
#     index_writer.commit()
#     index_writer.close()

# 读取查询并编码
queries = {}
with open('/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/library_source_code/queries_token.jsonl', 'r') as f:
    for line in tqdm(f, desc="Encoding queries", file=sys.stdout):
        query = json.loads(line)
        queries[query['id']] = {
            'text': query['text'],
            'embedding': encoder.encode(query['text']).to(device)
        }

# 对于每个查询，找到最接近的文档
results = {}
for q_id, query in tqdm(queries.items(), desc="Processing queries", file=sys.stdout):
    hits = searcher.search(query['embedding'].cpu().numpy(), k=1)
    if hits:
        top_hit = hits[0]
        results[q_id] = {'query_id': q_id, 'doc_id': top_hit.docid, 'score': top_hit.score}

# 输出结果到文件
output_file = '/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/library_source_code/similarity.jsonl'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")