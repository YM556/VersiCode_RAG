

from rdflib import Graph
from neo4j import GraphDatabase
import os

os.environ["NEO4J_URI"] = "neo4j://47.96.170.5:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Kangjz@123"



def create_node(tx,value:str):
    # import ipdb
    # ipdb.set_trace()
    if "DEPENDENCY" in value:
        value.replace("DEPENDENCY.","")
        result = tx.run("MERGE (n:Dependency {dependency:$text})",text=value)
        if result == None:
            print(value)
    # if "CODE" in value:
    #     value.replace("CODE.", "")
    #     tx.run("MERGE (n:CODE {code:$text})",text=value)
    # if "DESCRIPTION" in value:
    #     value.replace("DESCRIPTION.", "")
    #     tx.run("MERGE (n:Node {description:$text})",text=value)
    if "API" in value:
        value.replace("API.", "")
        tx.run("MERGE (n:API {api:$text})",text=value)



def create_relationship_contains(tx, dependency, api,rel):
    if rel == "CONTAINS":
        tx.run("MATCH (d:Dependency {dependency: $dependency}), (c:API {api: $api}) "
               "MERGE (d)-[r:CONTAINS]->(c)", dependency=dependency, api=api)
def create_relationship_explains(tx,description,code,rel):
    if rel == "EXPLAINS":
        tx.run("MATCH (d:Description {description: $description}), (c:Code {code: $code}) "
           "MERGE (d)-[r:EXPLAINS]->(c)", description=description, code=code)
def main():
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"],os.environ["NEO4J_PASSWORD"]))
    g = Graph()
    g.parse("output/api_graph.ttl", format="turtle")  # 使用你自己的 Turtle 文件路径
    with driver.session() as session:
        for subj, pred, obj in g:
            # import ipdb
            # ipdb.set_trace()
            # 添加节点（主语和宾语作为节点）
            session.write_transaction(lambda tx:create_node(tx,value=subj))
            session.write_transaction(lambda tx:create_node(tx,value=obj))

            if "CONTAINS" in str(pred):
                session.write_transaction(lambda tx:create_relationship_contains(tx,dependency=subj,api=obj,rel="CONTAINS"))

            # 添加关系（谓语作为关系）
            # if str(pred).endswith("EXPLAINS"):
            #     session.write_transaction(lambda tx:create_relationshop_explains(tx,description=obj,code=obj,rel="EXPLAINS") )
    driver.close()

if __name__ == "__main__":
    main()