import numpy as np
from neo4j import GraphDatabase
import gensim.models as g

# 加载模型
model_path = "output/node2vec.model"
model = g.Word2Vec.load(model_path)

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://47.96.170.5:7687", auth=("neo4j", "Kangjz@123"))
session = driver.session()

# 获取所有节点 ID
result = session.run('''MATCH (n)
RETURN COALESCE(n.dependency, n.api) AS node_id
  ''')


# 更新节点属性
for record in result:


    node_id = record["node_id"]
    node_id_str = str(node_id)  # 节点 ID 需要与模型中的键匹配

    if node_id_str in model.wv.key_to_index:
        node_embedding = model.wv[node_id_str]
        # 将 embedding 向量转换为字符串形式
        embedding_str = ','.join(map(str, node_embedding))

        # 使用 Cypher 更新节点
        session.run(
            """
            MATCH (n) 
                WHERE COALESCE(n.dependency, n.api) = $node_id
                SET n.embedding = $embedding

            """,
            node_id=node_id,
            embedding=embedding_str
        )

# 关闭会话和驱动
session.close()
driver.close()