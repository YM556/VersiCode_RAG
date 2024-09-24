"""
test llama3-70B token
"""
import json
from together import Together
import os
import tiktoken
import argparse
import ast
import pandas as pd
import numpy as np


import math


# encoding = tiktoken.get_encoding("gpt2")
max_tokens = 7000   #llama3-8b window 8k
# client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
# # model_name = "meta-llama/Llama-3-70b-chat-hf"
# model_name = "deepseek-ai/deepseek-coder-33b-instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--task",help="task chocies from token,line,block",default="token")
parser.add_argument("--with_rag", default=False, type=str,help="using rag or not",)
parser.add_argument("--model", type=str,default="meta-llama/Llama-3-70b-chat-hf")
parser.add_argument("--type", type=str,help="choices: add,deprecation,all")
parser.add_argument("--sample_num",type=int,default=200)
parser.add_argument("--loc",help="local remote")
parser.add_argument("--source",default="library_source_code")




args = parser.parse_args()
task = args.task

model_name = args.model
with_RAG = bool(args.with_rag)

if args.loc=="local":
    prefix_path = "/Volumes/kjz-SSD/Datasets"
else:
    prefix_path = '/root/autodl-tmp/'

print("With_RAG:{}".format(with_RAG))

n=10
ks= [1,10]
sample_num = args.sample_num
total_queries = 0
successful_predictions = 0
expected_outputs= []

results_10 = []
results_1 = []
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

def bulid_prompt_with_RAG(version, description, masked_code,extra_message) -> str:
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
    ###According to the embedding result and FAISS results, the most similar informations is below, wish you get the correct answer:
    {extra_message}
    ###response:
    """

    return prompt


json_path = os.path.join(prefix_path,'VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion',args.source,'{}_{}.json'.format(args.source,args.task))
similarity_file = os.path.join(prefix_path,'VersiCode-RAG/Reterival/output',args.source,args.task,'faiss_res.jsonl')
corpus_file = os.path.join(prefix_path,'VersiCode-RAG/Reterival/datasets',args.source,'corpus.jsonl')



def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    # if n - c < k: return 1.0
    # return 1.0 - np.prod(1.0 - k /
    #                      np.arange(n - c + 1, n + 1))
    if n - c < k:
        return 1.0
    score = 1 - (math.comb(n - c, k)) / (math.comb(n, k))
    return score


def compute_pass_at_k(predictions, expected_output, k):

    total_samples = len(predictions)
    pass_count = 0

    for preds, correct in zip(predictions, expected_output):
        # 检查前 k 个候选结果中是否包含正确答案
        if correct in preds[:k]:
            pass_count += 1
    return pass_at_k(len(predictions),pass_count,k)





def clean_prediction(text):
    clean_text = text.replace('<start>', '').replace('<end>', '')
    return clean_text

def reterive_msg(id:str,similarty_file,corpus_file):

    with open(similarty_file, 'r') as f_s:
        for line in f_s:
            data = json.loads(line)
            if data.get("query_id") == id:
                filter_df = pd.DataFrame([data])
                break
                # f.close()
    filter_dic = filter_df.to_dict()


    doc_id = filter_dic["doc_id"][0]


    with open(corpus_file,"r") as f_c:
        for line in f_c:
            data = json.loads(line)

            # import ipdb
            # ipdb.set_trace()
            if data.get("id") == doc_id:
                extra_msg = data["text"]
                break
                # f.close()

    # filter_df = df_c[df_c['id']==doc_id]
    # text = filter_df.to_dict(orient='records')[0]["text"]
    return extra_msg

def filter_data(json_path,type):
    with open(json_path, 'r', encoding='utf-8') as fr:
        lodict = json.load(fr)
    data_dict = lodict
    data_list = data_dict['data']

    # import ipdb
    # ipdb.set_trace()

    if type != 'all':
        df = pd.DataFrame(data_list)
        if type=='add':
            filter_df = df[df["type"]==("add" or "name_change_new")]
        elif type=='deprecation':
            filter_df = df[df["type"] == ("delete" or "name_change_old")]
        else:
            raise "Wrong type "
        data_list = filter_df.to_dict(orient='records')

    return lodict,data_list







data_dict,data_list = filter_data(json_path,args.type)



if len(data_list)<sample_num or sample_num==0:
    sample_num = len(data_list)
print(r"There are {} samples in this test,but we choose {} of them".format(len(data_list),sample_num))

import random

data_list = random.sample(data_list,sample_num)


for data in data_list[:sample_num]:
    predictions=[]
    if with_RAG==False:
        if "model_output" in data:
            print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
            continue
        try:

            print(f"predicting {data_list.index(data) + 1} ")
            version = data['dependency'] + data['version']  #   package == x.x.x
            description = data['description']   #   function description
            masked_code = data['masked_code']   #   masked code
            instruction = bulid_prompt(version, description, masked_code)
            truncated_text = truncate_text(instruction, max_tokens)
            for i in range(1):
                prediction = predict(truncated_text, model_name)
                data['model_output'] = prediction
                actual_list = ast.literal_eval(prediction)
                cleaned_list = [clean_prediction(item) for item in actual_list]

                # print(cleaned_list)
                predictions+=cleaned_list

            print("Per iter,the number of predictions is {}".format(len(predictions)))
            correct_number = sum(map(lambda x:x==data["answer"],predictions))

            results_10.append(pass_at_k(n,correct_number,10))
            results_1.append(pass_at_k(n,correct_number,1))

            pass_at_10 = np.array(results_10)
            pass_at_1 = np.array(results_1)

        except Exception as e:
            print(f"error：{e}")
            print("save current data")
            save_folder_path = os.path.join(r'../output/downstream/withoutRAG/{}'.format(task), model_name)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

            with open(save_json_path, 'w', encoding='utf-8') as fw:
                json.dump(data_dict, fw, indent=4, ensure_ascii=False)
            break
    else:
        if "model_output" in data:
            print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
            continue
        try:
            print(f"predicting {data_list.index(data) + 1} ")
            version = data['dependency'] + data['version']  # package == x.x.x
            description = data['description']  # function description
            masked_code = data['masked_code']  # masked code
            id = data['id']
            # import ipdb
            # ipdb.set_trace()

            extra_msg = reterive_msg(id,similarity_file,corpus_file)
            print("extra_msg")
            instruction = bulid_prompt_with_RAG(version, description, masked_code,extra_msg)

            truncated_text = truncate_text(instruction, max_tokens)

            for i in range(1):
                prediction = predict(truncated_text, model_name)
                # data['model_output'].append(prediction)
                actual_list = ast.literal_eval(prediction)
                cleaned_list = [clean_prediction(item) for item in actual_list]

                predictions+=cleaned_list


            print("Per iter,the number of predictions is {}".format(len(predictions)))
            correct_number = sum(map(lambda x:x==data["answer"],predictions))

            results_10.append(pass_at_k(n,correct_number,10))
            results_1.append(pass_at_k(n,correct_number,1))

            pass_at_10 = np.array(results_10)
            pass_at_1 = np.array(results_1)



        except Exception as e:
            print(f"error：{e}")
            print("save current data")
            save_folder_path = os.path.join(r'../output/downstream/withoutRAG/{}'.format(task), model_name)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

            with open(save_json_path, 'w', encoding='utf-8') as fw:
                json.dump(data_dict, fw, indent=4, ensure_ascii=False)
            break
    # break

if args.with_rag==True:
    part_path = "withRAG"
else:
    part_path ="withoutRAG"

save_folder_path = os.path.join(r'../output/library_source_code',args.task,part_path,model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])  #After an error occurs, use the address to continue predicting from where the error occurred


with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)

print("pass@1:{}".format(pass_at_1.mean()))
print("pass@10:{}".format(pass_at_10.mean()))



mission = args.model+'-'+part_path+'-'+args.type+'-'+str(sample_num)

output_dict  ={"task":mission,"pass@1:":pass_at_1.mean(),"pass@10:":pass_at_10.mean()}

with open(os.path.join(prefix_path,'VersiCode-RAG/Generation/output','pass@k.jsonl'),'a') as f:
    json.dump(output_dict,f,indent=4)





