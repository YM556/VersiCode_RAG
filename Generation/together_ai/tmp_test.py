"""
test llama3-70B token
"""
import json
import time

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
model_name = "meta-llama/Llama-3-70b-chat-hf"

parser = argparse.ArgumentParser()
# parser.add_argument("--task",help="task chocies from token,line,block",default="token")
parser.add_argument("--with_rag", default="False", type=str,help="using rag or not",)
parser.add_argument("--model", type=str,default="meta-llama/Llama-3-70b-chat-hf")
# parser.add_argument("--source", type=str,default="library_source_code",help="library_source_code, downstream_application_code,stackoverflow")
parser.add_argument("--sample_num",type=int,default=200)
parser.add_argument("--loc",help="local remote")
parser.add_argument("--corpus_dir",default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus",help="The directory contains all the library source code")
parser.add_argument("--library_version_style",type=str,default="same",help="retrieve message with matched version or not,same or remains")
parser.add_argument("--sample_source",type=str,default="mix"\
                    ,help="sample source, library_source_code,stackoverflow,downstream_application_code,\mixed means select from the other three")

args = parser.parse_args()
# task = args.task
with_RAG = args.with_rag
corpus_dir = args.corpus_dir
library_version_style = args.library_version_style
if args.loc=="local":
    prefix_path = "/Volumes/kjz-SSD/Datasets"
else:
    prefix_path = '/root/autodl-tmp/'

print("With_RAG:{}".format(with_RAG))
without_source=0

def reterive_msg(corpus_dir,dependency,version,library_version_style) -> list:
    source_code = []
    try:
        filepath = os.path.join(corpus_dir,dependency+".jsonl")
        if os.path.exists(filepath):
            print("Found source code {} {}".format(dependency,version))
            if version.startswith("=="):
                version = version[2:]
                with open(filepath,"r") as f:
                    for line in f:
                        data = json.loads(line)
                        if library_version_style=='same':
                            if data["version_num"] == version:
                                source_code.append({"file_path":data["file_path"],"source_code":data["source_code"]})
                        elif library_version_style=='remain':
                            if data["version_num"] != version:
                                source_code.append({"file_path": data["file_path"], "source_code": data["source_code"]})
            elif version.startswith(">="):
                 target_line = 0
                 version = version[2:]
                 with open(filepath,"r",encoding='utf-8') as f:
                     for idx,line in enumerate(f):
                         data = json.loads(line)
                         if data["version_num"] == version:
                             target_line = idx
                         if idx>=target_line and target_line!=0:
                            source_code.append({"file_path": data["file_path"], "source_code": data["source_code"]})
            elif version.startswith("<="):
                 version = version[2:]
                 with open(filepath, "r", encoding='utf-8') as f:
                     for idx, line in enumerate(f):
                         data = json.loads(line)
                         source_code.append({"file_path": data["file_path"], "source_code": data["source_code"]})
                         if data["version_num"] == version:
                             break
            else:
                raise "Wrong version number"

            return source_code
        else:
            print("Not find source code of {}".format(dependency))
            return source_code
    except Exception as e:
        print("Non library source contains")
        print(e)
        global without_source
        without_source+= 1
        return source_code


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


def bulid_prompt_with_rag(version, description, masked_code,source_code) -> str:
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
    prompt_source_code = [f"the file_path is {item['file_path']}, corresponding source code is {item['source_code']}" for item in source_code]

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
    Here I found the source code of the same version dependency: 
    {"".join(prompt_source_code)}
    ###response:
    """
    return prompt

def build_samples(sample_source:str,sample_num):
    sources = ["downstream_application_code","library_source_code","stackoverflow"]
    if sample_source in sources:
        sample_path = os.path.join(prefix_path,'VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/',sample_source,'{}_token.json'.format(sample_source))
        data_list,final_num = get_samples(sample_path,sample_num)
        return data_list,final_num
    else:
        nums=[]
        sample_paths = []
        data_list = []
        final_num = 0
        import random
        # downstream_application_code中只有80条sample
        nums.append(random.randint(1, 81))
        nums.append(random.randint(1, int(sample_num - nums[0]) - 1))
        nums.append(sample_num-nums[1]-nums[0])

        for source in sources:
            sample_paths.append(os.path.join(prefix_path,'VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/',source, '{}_token.json'.format(source)))
        for sample_path,num in zip(sample_paths,nums):
            _data_list,_final_num = get_samples(sample_path,num)
            final_num+=_final_num
            data_list += _data_list
    return data_list,final_num


def get_samples(sample_path,sample_num):
    with open(sample_path, 'r', encoding='utf-8')as fr:
        lodict = json.load(fr)
        _data_list = lodict['data']
        import random
        data_list = random.sample(_data_list, sample_num)
        return data_list, sample_num



# with open(json_path, 'r', encoding='utf-8')as fr:
#     lodict = json.load(fr)
# data_dict = lodict
# data_list = data_dict['data']

data_list,sample_num = build_samples(args.sample_source,args.sample_num)

for idx,data in enumerate(data_list):
    if "model_output" in data:
        print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
        continue
    try:
        start_time = time.time()
        print(f"predicting {data_list.index(data) + 1} ")
        version = data['dependency'] + data['version']  #   package == x.x.x
        description = data['description']   #   function description
        masked_code = data['masked_code']   #   masked code
        # import ipdb
        # ipdb.set_trace()
        if with_RAG=="True":
            source_code = reterive_msg(corpus_dir,data['dependency'],data['version'],library_version_style)
            if len(source_code)!=0:
                instruction = bulid_prompt_with_rag(version, description, masked_code,source_code)
            else:
                del data_list[idx]
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
        print("For one sample, it takes {:.3f} seconds".format(end_time-start_time))
    except Exception as e:
        del data_list[idx]
        print(f"error：{e}")
        print("save current data")
        print("current sample is {}".format(idx))
        data_dict = {"data":data_list[:idx]}
        save_folder_path = "/Users/qingyuanzii/Desktop/Versi-Code-Gen/VerisiCode-RAG/Generation/together_ai/tmp/token_results/"
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        save_json_path =os.path.join(save_folder_path,"test_token_source_code_rag.jsonl")

        with open(save_json_path, 'w', encoding='utf-8') as fw:
            json.dump(data_dict, fw, indent=4, ensure_ascii=False)
        break
    # break
data_dict = {"data":data_list}
save_folder_path = "/Users/qingyuanzii/Desktop/Versi-Code-Gen/VerisiCode-RAG/Generation/together_ai/tmp/token_results/"
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, "tmp_test.jsonl")
#After an error occurs, use the address to continue predicting from where the error occurred


with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)
print(save_json_path)
print("Completed sample number is {}".format(len(data_list)))




