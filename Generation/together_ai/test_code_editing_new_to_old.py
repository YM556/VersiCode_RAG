"""
test llama3-70B editing, new to old
"""
import json
from together import Together
import os
import tiktoken
# encoding = tiktoken.get_encoding("gpt2")
max_tokens = 7000   #llama3-8b window 8k
client = Together(api_key='159c9dc94810155413c9c4c7b2022eb232fc90e0d8f781d382604d029fd51b4b')
model_name = "meta-llama/Llama-3-70b-chat-hf"


def predict(text:str, model_name:str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        frequency_penalty=0.1,
        max_tokens=512,
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


def bulid_prompt(description, old_version, new_code, new_version) -> str:
    """
    construct prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """
    prompt = f"""
    You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, 
    including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified old version. 
    Your task is to refactor the code using the methods provided by the specified old version and return the refactored code. 
    Please note that you only need to return the refactored code and enclose it with <start> and <end>:
    ###Functionality description of the code
    {description}
    ###Dependency and origin version
    {new_version}
    ###Old version code
    {new_code}
    ###Dependency and old version
    {old_version}
    ###Refactored new code
    """

    return prompt


json_path = '/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/code_editing_new_to_old.json'

with open(json_path, 'r', encoding='utf-8') as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']

for data in data_list[:200]:
    print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
    old_version = data['dependency'] + data['old_version']  # package == x.x.x
    new_version = data['dependency'] + data['new_version']  # package == x.x.x
    description = data['description']  # function description
    new_code = data['new_code']

    instruction = bulid_prompt(description, old_version, new_code, new_version)
    prediction = predict(instruction, model_name)
    data['model_output'] = prediction

save_folder_path = os.path.join('../results_gen/code_editing/new_to_old/without_rag', model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8') as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)

