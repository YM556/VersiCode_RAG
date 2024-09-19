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
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


# encoding = tiktoken.get_encoding("gpt2")
max_tokens = 7000   #llama3-8b window 8k
client = Together(api_key='5ec70d43eb4422420ba0e09ca44d5d2a65bcb3661c31855584d4e90a2d228f54')
# # model_name = "meta-llama/Llama-3-70b-chat-hf"
# model_name = "deepseek-ai/deepseek-coder-33b-instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--task",help="task chocies from token,line,block",default="token")
parser.add_argument("--with_rag", default=False, type=bool,help="using rag or not",)
parser.add_argument("--model", type=str,default="meta-llama/Llama-3-70b-chat-hf")
parser.add_argument("--type", type=str,help="choices: add,general,deplicate,all")
parser.add_argument("--sample_num",type=int,default=300)

args = parser.parse_args()
task = args.task

model_name = args.model
with_RAG = args.with_rag


k=10
sample_num = args.sample_num
total_queries = 0
successful_predictions = 0
expected_outputs= []
predictions = []

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
max_length = model.config.max_position_embeddings  # The maximum context length of the model
print(f"max_length:{max_length}")


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
    ###extra message
    Here are some extra message can be refered:
    {extra_message}
    ###response:
    """

    return prompt


json_path = r'/root/autodl-tmp/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/library_source_code/library_source_code_{}.json'.format(task)
similarity_file = r'/root/autodl-tmp/VersiCode-RAG/Reterival/output/library_source_code/token/faiss_res.jsonl'
corpus_file = r'/root/autodl-tmp/VersiCode-RAG/Reterival/datasets/library_source_code/corpus.jsonl'


def compute_pass_at_k(predictions, expected_output, k):
    # import ipdb
    # ipdb.set_trace()
    total_samples = len(predictions)
    pass_count = 0

    for preds, correct in zip(predictions, expected_output):
        # 检查前 k 个候选结果中是否包含正确答案
        if correct in preds[:k]:
            pass_count += 1
    return pass_count / total_samples

def clean_prediction(text):
    clean_text = text.replace('<start>', '').replace('<end>', '')
    return clean_text

def reterive_msg(id:str,similarty_file,corpus_file):
    # 加载 JSONL 文件为 DataFrame
    df = pd.read_json(similarty_file, lines=True)
    filter_df = df[df['query_id'] == id]
    filter_dic = filter_df.to_dict(orient='records')
    if filter_df.empty:
        raise "Wrong id"

    doc_id = filter_dic[0]["doc_id"]

    # import ipdb
    # ipdb.set_trace()

    df_c = pd.read_json(corpus_file,lines=True)
    filter_df = df_c[df_c['id']==doc_id]
    text = filter_df.to_dict(orient='records')[0]["text"]
    return text

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
            filter_df = df[df["type"]=="add"]
        elif type=='general':
            filter_df = df[df["type"] == ("name_change_old"or"name_change_new")]
        elif type=='deplication':
            filter_df = df[df["type"] == "delete"]
        else:
            raise "Wrong type "
        data_list = filter_df.to_dict(orient='records')

    return lodict,data_list








data_dict,data_list = filter_data(json_path,args.type)



if len(data_list)<sample_num or sample_num==0:
    sample_num = len(data_list)
print(r"There are {} samples in this test,but we choose {} of them".format(len(data_list),sample_num))


data_list = random.sample(data_list,sample_num)
test_list = []
for data in data_list[:sample_num]:

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
            # truncated_text = truncate_text(instruction, max_tokens)
            # prediction = predict(truncated_text, model_name)
            # data['model_output'] = prediction

            # print("Prediction:{}, Answer:{}".format(prediction,data["answer"]))


            # expected_outputs.append(data["answer"])
            #
            # actual_list = ast.literal_eval(prediction)
            # cleaned_list = [clean_prediction(item) for item in actual_list]
            #
            # # print(cleaned_list)
            # predictions.append(cleaned_list)

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
            instruction = bulid_prompt_with_RAG(version, description, masked_code,extra_msg)

            # truncated_text = truncate_text(instruction, max_tokens)
            # prediction = predict(truncated_text, model_name)
            # data['model_output'] = prediction
            # data['model_output'] = prediction
            #
            # expected_outputs.append(data["answer"])
            #
            # actual_list = ast.literal_eval(prediction)
            # cleaned_list = [clean_prediction(item) for item in actual_list]
            #
            # # print(cleaned_list)
            # predictions.append(cleaned_list)

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
    encoded_inputs = tokenizer(instruction, return_tensors="pt")
    if encoded_inputs.input_ids.size(1) > max_length:
        encoded_inputs = {key: tensor[:, :max_length] for key, tensor in encoded_inputs.items()}

    instruction = tokenizer.decode(encoded_inputs['input_ids'][0], skip_special_tokens=True)

    test_list.append(instruction)

sampling_params = SamplingParams(n=100, temperature=0.8, top_p=0.95, max_tokens=64)

llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.9, swap_space=40)

outputs = llm.generate(test_list, sampling_params)

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

pass_at_k = compute_pass_at_k(predictions,expected_outputs,k)

print("pass@{} is {}".format(k,pass_at_k))

mission = args.model+'-'+part_path+'-'+args.type+'-'+str(sample_num)

output_dict  ={"task":mission,"pass@{}".format(k):pass_at_k}

with open('/root/autodl-tmp/VersiCode-RAG/Generation/output/library_source_code/pass@k.jsonl','a') as f:
    json.dump(output_dict,f,indent=4)





