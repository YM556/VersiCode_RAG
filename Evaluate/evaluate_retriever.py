"""
test llama3-70B token
"""

import time
import json

import chromadb
from sentence_transformers import SentenceTransformer

from together import Together
import os

import argparse

# encoding = tiktoken.get_encoding("gpt2")
max_tokens = 7000   #llama3-8b window 8k
client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
model_name = "meta-llama/Llama-3-70b-chat-hf"


parser = argparse.ArgumentParser()
parser.add_argument("--loc",help="local remote",default="remote")
parser.add_argument("--corpus_dir",default="VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus",help="The directory contains all the library source code")
parser.add_argument("--sample_num",type=int,default=200)

args = parser.parse_args()

total_time=0
sample_num = args.sample_num


if args.loc=="local":
    prefix_path = "/Volumes/kjz-SSD/Datasets"
else:
    prefix_path = '/root/autodl-tmp/'

corpus_dir = os.path.join(prefix_path,args.corpus_dir)

without_source=0

involve = 0
all_msg = 0
k=10


# 初始化 Chroma 客户端
_client = chromadb.Client()

def create_collection(base_dir,library_name,version,text,n_results=1):
    if "downstream_application_code" in base_dir:
        base_dir = '/root/autodl-tmp/VersiCode_Raw/VersiCode_Raw/downstream_application_code/project_specific_version_corpus'
    if version!="":
        file_path = os.path.join(base_dir,library_name,version+'.jsonl')
    else:
        file_path = os.path.join(base_dir, library_name + '.jsonl')
    if not os.path.exists(file_path):
        return [],0,0,0,0
    # 为每个文件创建一个 Collection dependency+version
    if version!= "":
        collection_name = library_name+'_'+version
    else:
        collection_name = library_name
    print("collection_name: {}".format(collection_name))

    collection = _client.get_or_create_collection(name=collection_name)
    collections = _client.list_collections()
    c_start_time = time.time()
    if not collection_name in collections:
        documents = []
        ids = []
        query_embeddings = []
        with open(file_path, 'r') as f:
            cnt = 0
            for idx,line in enumerate(f):
                item = json.loads(line)
                id = collection_name+'_'+str(idx)
                if "stackoverflow" in file_path:
                    source_code = item["question_answer"]
                else:
                    source_code = item["source_code"]
                if idx<1000:
                    documents.append(source_code)
                    ids.append(id)
                cnt+=1
        collection.add(documents=documents,ids=ids)
        print("Create Collection {} successfully".format(collection_name))
        c_end_time = time.time()
        # text = "This code defines a test class called BaseTest that inherits from absltest.TestCase. It also includes a test method called test_simple_functionality, which asserts that the method called system_under_test returns the value 1."
        e_start_time = time.time()
        query_embeddings.append(generate_embedding(text))
        e_end_time = time.time()
        r_start_time = time.time()
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
        r_end_time = time.time()
        retrieve_time = r_end_time-r_start_time
        create_collection_time = c_end_time-c_start_time
        embedding_time = e_end_time - e_start_time
        # print(results)
        return results["documents"][0],retrieve_time,create_collection_time,embedding_time,cnt


def generate_embedding(text):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text).tolist()
    return embedding


def reterive_msg(corpus_dir,dependency,version,text):
    source_code = []
    try:
        if "stackoverflow" in corpus_dir:
            version = ""
        else:
            if version.startswith("==") or version.startswith(">=") or version.startswith("<=") or version.startswith("~="):
                version = version[2:]
            else:
                print("Wrong Version number {}".format(version))
        return create_collection(corpus_dir,dependency,version,text,1)
    except Exception as e:
        print(e)
        raise "Wrong path!"





for name in ["library_source_code"]:
    input_path = os.path.join("/root/autodl-tmp/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/", name,
        "samples", "{}_token.json".format(name))

    with open(input_path,"r") as f:
        lodict = json.load(f)
    data_list = lodict["data"]
    for external_source in ["library_source_code","downstream_application_code","stackoverflow"]:
    # for external_source in ["downstream_application_code"]:
        sources = 0
        find_answer = 0
        total_time = 0
        res = []
        print("\n\n\nExternal Souces:{} \n\n\n".format(external_source))
        for idx,data in enumerate(data_list[:args.sample_num]):
            try:
                print(f"predicting {data_list.index(data) + 1} ")
                id = data["id"]
                version = data['dependency'] + data['version']  #   package == x.x.x
                description = data['description']   #   function description
                masked_code = data['masked_code']   #   masked code
                answer = data['answer']

                # document_msg.keys() ['ids', 'distances', 'metadatas', 'embeddings', 'documents', 'uris', 'data', 'included']
                corpus_dir = '/root/autodl-tmp/VersiCode_Raw/VersiCode_Raw/'+external_source+'/version_corpus'
                document_msg, retrieve_time, c_time, e_time, item_num = reterive_msg(corpus_dir,data['dependency'],data['version'],description)
                total_time+=retrieve_time+c_time+e_time
                count_number = 0
                if len(document_msg)>0:
                    extra_msg = document_msg[0]
                    if answer in extra_msg:
                        find_answer+=1
                        count_number = extra_msg.count(answer)

                    res.append({"query_id": id, "answer": answer, "dependency": version,
                                "contain number": count_number,
                                "retreive_time": retrieve_time,
                                "create_collection_time":c_time,
                                "embedding_time":e_time,
                                "item_number":item_num})
                    sources+=1
                else:
                    continue
            except Exception as e:
                # del data_list[idx]
                print(f"error：{e}")

            # break
        data_dict={"data":res}

        save_folder_path = "./tmp/token_results/retriever/"
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        save_json_path = os.path.join(save_folder_path,r"{}_from_{}.jsonl".format(name,external_source))
        #After an error occurs, use the address to continue predicting from where the error occurred

        data_dict = {"data":res}
        with open(save_json_path, 'w', encoding='utf-8')as fw:
            json.dump(data_dict, fw, indent=4, ensure_ascii=False)

        print(save_json_path)
        print("Completed sample number is {}".format(len(data_list)))
        print("total seconds: {}".format(total_time))

        print("{} of {} sources found answer in sources ".format(find_answer,sources))
        if sources!=0:
            print(find_answer / sources)
            print("Average time per sample is {}".format(total_time / sources))
