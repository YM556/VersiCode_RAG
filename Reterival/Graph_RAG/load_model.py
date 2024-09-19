import gensim.models as g

# 加载模型
model_path = "output/node2vec.model"
model = g.Word2Vec.load(model_path)

# 检查模型的一些基本信息
print("Model details:")
print("Dimensions:", model.vector_size)
print("Vocab size:", len(model.wv.index_to_key))

# 获取节点的embedding
node_id = "file:///Users/qingyuanzii/Desktop/Versi-Code-Gen/VerisiCode-RAG/Reterival/Graph_RAG/output/API.torch.jit.verify"  # 假设"A"是你的图中的一个节点
if node_id in model.wv.key_to_index:
    node_embedding = model.wv[node_id]
    print(f"Embedding for node {node_id}: {node_embedding}")
else:
    print(f"Node {node_id} not found in the model vocabulary.")