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
client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
# # model_name = "meta-llama/Llama-3-70b-chat-hf"
# model_name = "deepseek-ai/deepseek-coder-33b-instruct"




parser = argparse.ArgumentParser()
parser.add_argument("--task",help="task chocies from token,line,block",default="token")
parser.add_argument("--with_rag", default=False, type=bool,help="using rag or not",)
parser.add_argument("--model", type=str,default="meta-llama/Llama-3-70b-chat-hf")
parser.add_argument("--source", type=str,default="library_source_code",help="library_source_code, downstream_application_code,stackoverflow")
parser.add_argument("--sample_num",type=int,default=200)
parser.add_argument("--loc",help="local remote")

args = parser.parse_args()
task = args.task

model_name = args.model
with_RAG = bool(args.with_rag)

print("With_RAG:{}".format(with_RAG))

if args.loc=="local":
    prefix_path = "/Volumes/kjz-SSD/Datasets"
else:
    prefix_path = '/root/autodl-tmp/'



json_path = os.path.join(prefix_path,'VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion',args.source,'{}_{}.json'.format(args.source,args.task))
similarity_file = os.path.join(prefix_path,'VersiCode-RAG/Reterival/output',args.source,args.task,'faiss_res.jsonl')
corpus_file = os.path.join(prefix_path,'VersiCode-RAG/Reterival/datasets',args.source,'corpus.jsonl')

# json_path = os.path.join('/root/autodl-tmp/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion',args.source,'{}_{}.json'.format(args.source,task))
# similarity_file = os.path.join('/root/autodl-tmp/VersiCode-RAG/Reterival/output',args.source,args.task,'faiss_res.jsonl')
# corpus_file = r'/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/downstream_application/corpus.jsonl'


n=10
ks= [1,3,10]
sample_num = args.sample_num
total_queries = 0
successful_predictions = 0
expected_outputs= []

results_10 = []
results_1 = []
results_3 = []
list_c = [0,0,0,0,0,0,0,0,0,0,0]


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    # if n - c < k: return 1.0
    # return 1.0 - np.prod(1.0 - k /
    #                      np.arange(n - c + 1, n + 1))
    # if n - c < k:
    #     return 1.0
    # score = 1 - (math.comb(n - c, k)) / (math.comb(n, k))
    if n-c<k:
        return 1.0
    print("n:{},c:{},k:{}".format(n, c, k))
    score = 1 - (math.comb(n - c, k)) / (math.comb(n, k))

    return score

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


def bulid_prompt_with_RAG(version, description, masked_code, extra_message) -> str:
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
        #According to the embedding result and FAISS results, the most similar information is below, wish you get the correct answer:
        {extra_message}
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
    #According to the embedding result and FAISS results, the most similar information is below, wish you get the correct answer:
    {extra_message}
    ###response:
    """

    return prompt


def clean_prediction(output):
    try:
        if "<start>" in output and "<end>" in output:
            start_index = output.find("<start>") + len("<start>")
            end_index = output.find("<end>")
            text = output[start_index:end_index].replace('```python', '').replace('```', '')
            return text
    except:
        import ipdb
        ipdb.set_trace()
        print(output)


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
            if data.get("id") == doc_id:
                extra_msg = data["text"]
                break
                # f.close()

    # filter_df = df_c[df_c['id']==doc_id]
    # text = filter_df.to_dict(orient='records')[0]["text"]
    return extra_msg

with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
    data_dict = lodict
    _data_list = data_dict['data']



if len(_data_list)<sample_num or sample_num==0:
    sample_num = len(_data_list)
print(r"There are {} samples in this test,but we choose {} of them".format(len(_data_list),sample_num))

import random
data_list = random.sample(_data_list,sample_num)

for data in data_list[:sample_num]:

    if with_RAG==False:
        if "model_output" in data:
            print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
            continue
        try:
            print(f"predicting {data_list.index(data) + 1} ")
            version = data['dependency'] + data['version']  # package == x.x.x
            description = data['description']  # function description
            masked_code = data['masked_code']  # masked code
            id = data['id']

            instruction = bulid_prompt(version, description, masked_code)
            truncated_text = truncate_text(instruction, max_tokens)
            predictions = []
            for i in range(1):

                _prediction = predict(truncated_text, model_name)
                prediction = eval(_prediction)
                # import ipdb
                # ipdb.set_trace()
                cleaned_list = [clean_prediction(item) for item in prediction]
                predictions += cleaned_list
                print("Per iter,the number of predictions is {}".format(len(predictions)))
                correct_number = sum(map(lambda x: x == data["answer"], predictions))
                list_c[correct_number] += 1
                print(predictions)
                print(data["answer"])
                print(correct_number)

                tmp_score1 = pass_at_k(n, correct_number, 1)
                tmp_score3 = pass_at_k(n, correct_number, 3)
                tmp_score10 = pass_at_k(n, correct_number, 10)

                results_1.append(tmp_score1)
                results_3.append(tmp_score3)
                results_10.append(tmp_score10)


                pass_at_10 = np.array(results_10)
                pass_at_3 = np.array(results_3)
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
            extra_msg = reterive_msg(id,similarity_file,corpus_file)
            # print(extra_msg)
            instruction = bulid_prompt_with_RAG(version, description, masked_code,extra_msg)
            truncated_text = truncate_text(instruction, max_tokens)
            predictions = []
            for i in range(1):
                _prediction = predict(truncated_text, model_name)
                prediction = eval(_prediction)
                cleaned_list = [clean_prediction(item) for item in prediction]
                predictions += cleaned_list

                print("Per iter,the number of predictions is {}".format(len(predictions)))
                correct_number = sum(map(lambda x:x==data["answer"],predictions))
                print(predictions)
                print(data["answer"])
                print(correct_number)

                tmp_score1 = pass_at_k(n, correct_number, 1)
                tmp_score3 = pass_at_k(n, correct_number, 3)
                tmp_score10 = pass_at_k(n, correct_number, 10)

                results_1.append(tmp_score1)
                results_3.append(tmp_score3)
                results_10.append(tmp_score10)

                pass_at_10 = np.array(results_10)
                pass_at_3 = np.array(results_3)
                pass_at_1 = np.array(results_1)
                #
                # print("correct_number:{}".format(correct_number))
                # print(pass_at_10)


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

save_folder_path = os.path.join(r'../output/{}/withoutRAG/{}'.format(args.source,args.task), model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])  #After an error occurs, use the address to continue predicting from where the error occurred


with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)

print("pass@1:{}".format(pass_at_1.mean()))
print("pass@3:{}".format(pass_at_3.mean()))
print("pass@10:{}".format(pass_at_10.mean()))

print(sum(results_1)/len(results_1))
print(sum(results_3)/len(results_3))
print(sum(results_10)/len(results_10))


mission = args.model+'-'+part_path+'-'+'-'+str(sample_num)

output_dict  ={"task":mission,"pass@1:":pass_at_1.mean(),"pass@3":pass_at_3.mean(),"pass@10:":pass_at_10.mean()}

with open(os.path.join(prefix_path,'VersiCode-RAG/Generation/output','pass@k.jsonl'),'a') as f:
    json.dump(output_dict,f,indent=4)
print("list_c:")
print(list_c)

