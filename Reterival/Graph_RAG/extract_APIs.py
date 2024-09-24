import ast
import json
from rdflib import Graph, URIRef, Literal, Namespace
from tqdm import tqdm


class JsonReader:
    def __init__(self):
        self.json_reader_list = []
def load_json(json_file):
    jsonreader = JsonReader()
    with open(json_file,"r") as f:
        for line in f:
            item = json.loads(line)
            jsonreader.json_reader_list.append(item)
    return jsonreader.json_reader_list

class APICallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.api_calls = []

    def get_full_attribute(self, node):
        """Recursive function to retrieve full attribute path"""
        full_api = []
        while isinstance(node, ast.Attribute):
            full_api.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            full_api.append(node.id)
        full_api.reverse()
        return ".".join(full_api)

    def visit_Call(self, node):
        """Handles function or method calls"""
        # Extract full API only for function/method calls
        if isinstance(node.func, ast.Attribute):
            full_api = self.get_full_attribute(node.func)
            self.api_calls.append(full_api)
        elif isinstance(node.func, ast.Name):
            # This handles direct function calls like 'print()'
            self.api_calls.append(node.func.id)
        self.generic_visit(node)

def extract_api_calls(code,dep_name):
    """Parses the code and extracts API calls"""
    try:
        tree = ast.parse(code)
    except:
        return []
    visitor = APICallVisitor()
    visitor.visit(tree)
    api_calls = visitor.api_calls
    # api_calls = [x for x in visitor.api_calls if x.startswith(dep_name)]
    api_calls = list(set(list(api_calls)))
    return api_calls

def reformat_prefix(item, prf: str):
    return prf + str(item)


if __name__ == "__main__":
    data_list = load_json(json_file="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/corpus_jsonl/torch.jsonl")
    g = Graph()

    for idx,line in tqdm(enumerate(data_list),total=len(data_list)):

        dep_name = line["library_name"] + line["version_num"]
        dependency = reformat_prefix(dep_name, "DEPENDENCY.")
        APIs = extract_api_calls(str(line["source_code"]),line["library_name"])
        if len(APIs) == 0:
            continue
        for _api in APIs:
            api = reformat_prefix(_api,"API.")
            exists = (URIRef(dependency),URIRef("CONTAINS"),URIRef(api)) in g
            if not exists:
                g.add((URIRef(dependency),URIRef("CONTAINS"),URIRef(api)))
    g.serialize(destination="output/torch_api.ttl", format="turtle")





