"""
test llama3-70B token
"""

import json

import time
import json
import chromadb
from sentence_transformers import SentenceTransformer

from together import Together
import os
import tiktoken
import argparse

# encoding = tiktoken.get_encoding("gpt2")
max_tokens = 7000   #llama3-8b window 8k
client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
model_name = "meta-llama/Llama-3-70b-chat-hf"


parser = argparse.ArgumentParser()
# parser.add_argument("--task",help="task chocies from token,line,block",default="token")
parser.add_argument("--with_rag", default="False", type=str,help="using rag or not",)
parser.add_argument("--model", type=str,default="meta-llama/Llama-3-70b-chat-hf")
# parser.add_argument("--source", type=str,default="library_source_code",help="library_source_code, downstream_application_code,stackoverflow")
parser.add_argument("--sample_num",type=int,default=200)
parser.add_argument("--loc",help="local remote")
parser.add_argument("--corpus_dir",default="VersiCode_Raw/VersiCode_Raw/stackoverflow/version_corpus",help="The directory contains all the library source code")
parser.add_argument("--library_version_style",type=str,default="same",help="retrieve message with matched version or not,same or remains")
parser.add_argument("--sample_source",type=str,default="library_source_code"\
                    ,help="sample source, library_source_code,stackoverflow,downstream_application_code,\mixed means select from the other three")

args = parser.parse_args()
with_RAG = args.with_rag

total_time=0

library_version_style = args.library_version_style
if args.loc=="local":
    prefix_path = "/Volumes/kjz-SSD/Datasets"
else:
    prefix_path = '/root/autodl-tmp/'

corpus_dir = os.path.join(prefix_path,args.corpus_dir)
source_code_prompt_length =10
print("With_RAG:{}".format(with_RAG))
without_source=0

involve = 0
all_msg = 0


# 初始化 Chroma 客户端
_client = chromadb.Client()

def create_collection(base_dir,library_name,text,n_results=1,):
    # import ipdb
    # ipdb.set_trace()
    file_path = os.path.join(base_dir,library_name+'.jsonl')
    # 为每个文件创建一个 Collection dependency+version
    collection_name = library_name
    print("collection_name: {}".format(collection_name))
    collection = _client.get_or_create_collection(name=collection_name)
    collections = _client.list_collections()
    if not collection_name in collections:
        documents = []
        ids = []
        query_embeddings = []
        with open(file_path, 'r') as f:
            for idx,line in enumerate(f):
                if idx<1000:
                    item = json.loads(line)
                    id = collection_name+'_'+str(idx)
                    source_code = item["question_answer"]
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
        # import ipdb
        # ipdb.set_trace()
        print(results["documents"][0][0])
        return results["documents"][0][0]


def generate_embedding(text):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    embedding = model.encode(text).tolist()
    return embedding


def reterive_msg(corpus_dir,dependency,text) -> list:
    source_code = []
    try:
        filepath = os.path.join(corpus_dir,dependency+".jsonl")
        if os.path.exists(filepath):
            r_start_time = time.time()
            _source_code = create_collection(corpus_dir,dependency,text,1)
            r_end_time = time.time()
            source_code.append(_source_code)
            retrieve_time = r_end_time-r_start_time
        return source_code,retrieve_time
    except Exception as e:
        print("Non library source contains")
        print(e)
        global without_source
        without_source+= 1
        return source_code,0


def truncate_text(text, max_tokens):
    # obtain tokenizer
    encoding = tiktoken.get_encoding("gpt2")
    disallowed_special = ()
    tokens = encoding.encode(text, disallowed_special=disallowed_special)
    print(len(tokens))

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    truncated_text = encoding.decode(tokens)

    return truncated_text

def predict(text:str, model_name:str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        frequency_penalty=0.1,
        max_tokens=64,
        logit_bias=None,
        logprobs=None,
        n=10,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        temperature=0.8,
        top_p=0.95
    )
    choices_list = response.choices

    ans_list = []
    for c in choices_list:
        content = c.message.content
        if "," in content:
            content = content.split(',')[0]
        ans_list.append(content)
    final_ans = str(ans_list)


    return final_ans
def bulid_prompt(version, description, masked_code) -> str:
    """
    construct prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """
    prompt = f"""
    You are a professional Python programming engineer, and I will give you a code snippet where function names are masked and represented as<mask>in the code. 
    There may be multiple <mask>, and all the blocked content in these <mask> is the same. 
    I will provide a functional description of this code, the dependency package to which the function belongs and the version of the dependency package.
    What you need to do is infer what the masked function name is based on this information. You only need to return one content, not every<mask>.
    Please note that you only need to return one function name and do not need to return any other redundant content, and the response is enclosed by <start> and <end>.Here is an example:
    ###code snippet:
    outputs = llm.<mask>(prompts, sampling_params)
    ###Function Description:
    This code passes prompts and parameters to the model to obtain the output result of the model.
    ###dependeny and version:
    vllm==0.3.3
    ###response:
    <start>generate<end>

    ###code snippet:
    {masked_code}
    ###Function Description:
    {description}
    ###dependeny and version:
    {version}
    ###response:
    """

    return prompt


def bulid_prompt_with_rag(dependency, description, masked_code,extra_msg) -> str:
    """
    construct prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """
    # import ipdb
    # ipdb.set_trace()
    prompt = f"""
    You are a professional Python programming engineer, and I will give you a code snippet where function names are masked and represented as<mask>in the code. 
    There may be multiple <mask>, and all the blocked content in these <mask> is the same. 
    I will provide a functional description of this code, the dependency package to which the function belongs and the version of the dependency package.
    What you need to do is infer what the masked function name is based on this information. You only need to return one content, not every<mask>.
    Please note that you only need to return one function name and do not need to return any other redundant content, and the response is enclosed by <start> and <end>.Here is an example:
    ###code snippet:
    outputs = llm.<mask>(prompts, sampling_params)
    ###Function Description:
    This code passes prompts and parameters to the model to obtain the output result of the model.
    ###dependeny and version:
    vllm==0.3.3
    ###response:
    <start>generate<end>

    ###code snippet:
    {masked_code}
    ###Function Description:
    {description}
    ###dependeny
    {dependency}
    Here I found the most relevant information from stackoverflow, within same dependency. I hope that will be helpful: 
    {extra_msg}
    ###response:
    """
    return prompt

def build_samples(sample_source:str,sample_num):
    sources = ["downstream_application_code", "library_source_code", "stackoverflow"]
    if sample_source in sources:
        sample_path = os.path.join(prefix_path,'VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/',sample_source,'samples','{}_token.json'.format(sample_source))
        data_list,final_num = get_samples(sample_path,sample_num)
        return data_list,final_num
    else:
        raise "Wrong sample sources"


def get_samples(sample_path,sample_num):
    with open(sample_path, 'r', encoding='utf-8')as fr:
        lodict = json.load(fr)
        _data_list = lodict['data']
        data_list = _data_list[:sample_num]
        return data_list, sample_num

data_list,sample_num = build_samples(args.sample_source,args.sample_num)

retrieve_time_list = []
total_time_list = []
for idx,data in enumerate(data_list[:sample_num]):
    if "model_output" in data:
        print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
        continue
    try:

        print(f"predicting {data_list.index(data) + 1} ")
        version = data['dependency'] + data['version']  #   package == x.x.x
        description = data['description']   #   function description
        masked_code = data['masked_code']   #   masked code
        code = data['code']
        if with_RAG=="True":
            start_time = time.time()
            source_code,retrieve_time = reterive_msg(corpus_dir, data['dependency'], description)
            if len(source_code)!=0:
                if data["answer"] in source_code:
                    involve+=1
                all_msg+=1
                extra_msg = source_code[0]
                if code in extra_msg:
                    extra_msg.replace(code,"<masked>")
                instruction = bulid_prompt_with_rag(version, description, masked_code,extra_msg)
                print("source_code length is {}".format(len(source_code)))
            else:
                print("source_code length is {}".format(len(source_code)))
                continue
        else:
            instruction = bulid_prompt(version, description, masked_code)

        truncated_text = truncate_text(instruction, max_tokens)
        prediction = predict(truncated_text, model_name)
        print("prediction: " + prediction)
        print("answer: "+data["answer"])
        data['model_output'] = prediction
        end_time = time.time()
        retrieve_time_list.append(retrieve_time)
        total_time_list.append(end_time-start_time)
        print("For one sample, it takes {:.3f} seconds".format(end_time-start_time))

        print("data_list length: {}".format(len(data_list)))
    except Exception as e:
        # del data_list[idx]
        print(f"error：{e}")
        print("save current data")
        print("current sample is {}".format(idx))
        data_dict = {"data":data_list[:idx]}
        save_folder_path = "./tmp/token_results/"
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        save_json_path =os.path.join(save_folder_path,"test_token_source_code_rag.jsonl")

        with open(save_json_path, 'w', encoding='utf-8') as fw:
            json.dump(data_dict, fw, indent=4, ensure_ascii=False)
        break
    # break
data_dict = {"data":data_list}


save_folder_path = "./tmp/token_results/"
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, "token_rag_stackoveflow.jsonl")
#After an error occurs, use the address to continue predicting from where the error occurred


with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)
print(save_json_path)
print("Completed sample number is {}".format(len(data_list)))
print("total seconds: {}".format(total_time))
print("Average_time:{}".format(sum(total_time_list)/len(total_time_list)))
print("Average_retrieve_time: {}".format(sum(retrieve_time_list)/len(retrieve_time_list)))




