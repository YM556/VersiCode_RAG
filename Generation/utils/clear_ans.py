"""
Clear the<start>and<end>generated by the model in inference
"""

import json
import os

model_name = 'Llama-3-70b-chat-hf'

result_path = f'../../dataset/final_dataset/other_language_data_finalv2_result/{model_name}/javascript_test_block.json' #改 token,line,block


with open(result_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']

for data in data_list:
    temp_list = []
    model_output_list = eval(data['model_output'])  #改
    for output in model_output_list:

        if "<start>" in output and "<end>" in output:
            start_index = output.find("<start>") + len("<start>")
            end_index = output.find("<end>")
            content = output[start_index:end_index].replace('```python', '').replace('```', '')
        else:
            content = "no_answer"

        temp_list.append(content)

    data['model_output_token_clear'] = str(temp_list)   #change token,line,block

with open(result_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)