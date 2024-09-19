from rdflib import Graph

# 创建一个 Graph 对象
g = Graph()

# 读取 RDF 数据文件（例如 Turtle 格式）
g.parse("output/api_graph.ttl", format="turtle")
