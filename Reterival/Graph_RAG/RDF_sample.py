from rdflib import Graph, URIRef, Literal, Namespace

import json

# Load data from JSON file
with open("/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/torch_new_to_old.json", "r") as f:
    lodic = json.load(f)
    data_list = lodic["data"]

# Create an RDF graph
g = Graph()
EX = Namespace("library_token")

# Helper function to create full URIs
def reformat_prefix(item, prf: str):
    return prf + str(item)

# Process each item in the data list
for line in data_list:
    old_dep = reformat_prefix(line["dependency"] + line["old_version"][2:],"DEPENDENCY.")
    old_api = reformat_prefix(line["old_name"],"API.")
    new_dep = reformat_prefix(line["dependency"] + line["new_version"][2:],"DEPENDENCY.")
    new_api = reformat_prefix(line["new_name"],"API.")


    # Add triples to the graph
    # print(Literal(code))
    g.add((URIRef(old_dep), URIRef("CONTAINS"),URIRef(old_api)))
    g.add((URIRef(new_dep), URIRef("CONTAINS"),URIRef(new_api)))
    g.add((URIRef(old_api), URIRef("ALIGNS"),URIRef(new_api)))


# Serialize the graph to Turtle format
g.serialize(destination="output/torch_version.ttl", format="turtle")

