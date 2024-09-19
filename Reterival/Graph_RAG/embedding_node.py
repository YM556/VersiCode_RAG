import rdflib
from rdflib import Graph
import networkx as nx

# 读取TTL文件
g = Graph()
g.parse("output/torch_api.ttl", format="turtle")

# 构造图数据结构
G = nx.Graph()

# 遍历RDF三元组
for s, p, o in g.triples((None, None, None)):
    if isinstance(s, rdflib.term.URIRef) and isinstance(o, rdflib.term.URIRef):
        G.add_edge(str(s), str(o))

from node2vec import Node2Vec

# 创建Node2Vec对象
node2vec = Node2Vec(
    G,
    dimensions=128,  # 嵌入向量的维度
    walk_length=10,  # 每次随机游走的长度
    num_walks=100,   # 每个节点开始的随机游走次数
    p=1,             # 回到前一个节点的概率
    q=1,             # 转移到远离前一个节点的概率
    workers=4        # 并发工作的数量
)

# 生成随机游走序列
walks = node2vec.walks
import gensim.models as g

# 使用gensim训练模型
model = g.Word2Vec(
    walks,
    vector_size=384,  # 嵌入向量的维度
    window=5,          # 上下文窗口大小
    min_count=0,       # 忽略总频次低于此值的所有单词
    sg=1,              # 使用skip-gram模型（而不是CBOW）
    workers=4          # 并发工作的数量
)

# 获取节点的embedding
node_embedding = model.wv[str(list(G.nodes)[0])]

# 输出节点的embedding
print(f"Embedding for node {list(G.nodes)[0]}: {node_embedding}")
# 保存模型
model.save("output/node2vec.model")

# 加载模型
loaded_model = g.Word2Vec.load("output/node2vec.model")

