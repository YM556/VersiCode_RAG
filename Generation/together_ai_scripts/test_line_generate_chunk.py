"""
test llama3-70B line
"""
import json
from together import Together
import os
import tiktoken
import pandas as pd

import argparse

# encoding = tiktoken.get_encoding("gpt2")
max_tokens = 7000   #llama3-8b window 8k
client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')


parser = argparse.ArgumentParser()
parser.add_argument("--task",help="task chocies from token,line,block",default="line")
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

n=50
ks= [1,3,10]
sample_num = args.sample_num
total_queries = 0
successful_predictions = 0
expected_outputs= []

sum_ISM = 0
sum_PM = 0


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
        max_tokens=128,
        logit_bias=None,
        logprobs=None,
        n=6,
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
    prompt = f'''
            You will act as a professional Python programming engineer, and I will provide a code snippet where a certain line in the code will be masked and represented as<mask>.
            I will provide a functional description related to this code segment, the dependency packages related to this line of code, and the versions of the dependency packages.
            You need to infer the masked line of code based on this information. Note that you only need to return one line of code, and the line is the response you infer.
            Please be careful not to return the information I provided, only the content of the response needs to be returned Enclose that line of code with tags <start> and <end>. Here is an example:

            ###code snippet：
            for output in outputs:
                prompt = output.prompt
                <mask>
                print("Prompt,Generated text")
            ###Function Description：
            The function of this code is to print the results predicted by calling the model using vllm.
            ###dependeny and version：
            vllm==0.3.3
            ###response:
            <start>generated_text = output.outputs[0].text<end>

            ###code snippet：
            {masked_code}
            ###Function Description：
            {description}
            ###dependeny and version:
            {version}
            ###response:

        '''
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
    prompt = f'''
            You will act as a professional Python programming engineer, and I will provide a code snippet where a certain line in the code will be masked and represented as<mask>.
            I will provide a functional description related to this code segment, the dependency packages related to this line of code, and the versions of the dependency packages.
            You need to infer the masked line of code based on this information. Note that you only need to return one line of code, and the line is the response you infer.
            Please be careful not to return the information I provided, only the content of the response needs to be returned Enclose that line of code with tags <start> and <end>. Here is an example:

            ###code snippet：
            for output in outputs:
                prompt = output.prompt
                <mask>
                print("Prompt,Generated text")
            ###Function Description：
            The function of this code is to print the results predicted by calling the model using vllm.
            ###dependeny and version：
            vllm==0.3.3
            ###response:
            <start>generated_text = output.outputs[0].text<end>

            ###code snippet：
            {masked_code}
            ###Function Description：
            {description}
            ###dependeny and version:
            {version}
            #According to the embedding results and FAISS results, the most similar information is added below, wish you could get the correct answer:
            {extra_message}
            ###response:

        '''
    return prompt

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

# json_path = '../dataset/final_dataset/generate_line_result_data2024_5_14v2/gpt-4o/random_line_data_2024_5_14v2.json'  #After an error occurs, use the address to continue predicting from where the error occurred


with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']

if len(data_list)<sample_num or sample_num==0:
    sample_num = len(data_list)
print(r"There are {} samples in this test,but we choose {} of them".format(len(data_list),sample_num))

#   逐条预测
for data in data_list[:sample_num]:
    if with_RAG == False:
        print("without rag")
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
            prediction = predict(truncated_text, model_name)
            data['model_output'] = prediction
        except Exception as e:
            print(f"error：{e}")
            print("save current data")
            save_folder_path = os.path.join('./line', model_name)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

            with open(save_json_path, 'w', encoding='utf-8') as fw:
                json.dump(data_dict, fw, indent=4, ensure_ascii=False)
            break
        # break
    else:
        print("with rag")
        if "model_output" in data:
            print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
            continue
        try:
            print(f"predicting {data_list.index(data) + 1} ")
            version = data['dependency'] + data['version']  # package == x.x.x
            description = data['description']  # function description
            masked_code = data['masked_code']  # masked code
            id = data["id"]
            extra_msg = reterive_msg(id,similarity_file,corpus_file)

            instruction = bulid_prompt_with_RAG(version, description, masked_code,extra_msg)
            truncated_text = truncate_text(instruction, max_tokens)
            prediction = predict(truncated_text, model_name)

            data['model_output'] = prediction
        except Exception as e:
            print(f"error：{e}")
            print("save current data")
            save_folder_path = os.path.join('./line',model_name)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

            with open(save_json_path, 'w', encoding='utf-8') as fw:
                json.dump(data_dict, fw, indent=4, ensure_ascii=False)
            break
        # break





save_folder_path = os.path.join('./line',model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)

print(save_json_path)


