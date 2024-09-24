import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# 初始化 Chroma 客户端
client = chromadb.Client()

def create_collection(base_dir,library_name,version,text,n_results=1):
    file_path = os.path.join(base_dir,library_name,version+'.jsonl')
    # 为每个文件创建一个 Collection dependency+version
    collection_name = library_name+'_'+version
    print("collection_name: {}".format(collection_name))
    collection = client.get_or_create_collection(name=collection_name)
    collections = client.list_collections()
    if not collection_name in collections:
        documents = []
        ids = []
        query_embeddings = []
        with open(file_path, 'r') as f:
            for idx,line in enumerate(f):
                item = json.loads(line)
                id = collection_name+'_'+str(idx)
                source_code = item["source_code"]
                documents.append(source_code)
                ids.append(id)
        collection.add(documents=documents,ids=ids)
        print("Create Collection {} successfully".format(collection_name))
        # text = "This code defines a test class called BaseTest that inherits from absltest.TestCase. It also includes a test method called test_simple_functionality, which asserts that the method called system_under_test returns the value 1."
        query_embeddings.append(generate_embedding(text))
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
        return results["documents"][0][0]


def generate_embedding(text):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text).tolist()
    return embedding
