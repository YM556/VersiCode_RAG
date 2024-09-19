"""
pass@k Indicator to evaluate the ability of token level
"""
import argparse
import json
import os
import math
import sys

model_name = 'Llama-3-70b-chat-hf'

# parser = argparse.ArgumentParser()
# parser.add_argument("--result_path",type=str)
# args = parser.parse_args()

token_json_path = sys.argv[1]
sample_num = int(sys.argv[2])


# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/docstring.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/respository.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/stackoverflow.json'
def compute_score_k(answer:str, model_output:list, k:int):

    c = 0
    n = len(model_output)
    for output in model_output:
        if answer == output:
            c += 1
    if n-c<k:
        return 1.0
    print("n:{},c:{},k:{}".format(n,c,k))
    score = 1 - (math.comb(n - c, k))/(math.comb(n, k))

    return score



with open(token_json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data = lodict

data_list = data['data']
data_len = len(data_list)
score_list_1 = []
score_list_3 = []
score_list_10 = []

ks=[1,3,10]

for d in data_list[:sample_num]:
    answer = d['answer']
    model_output_list = eval(d['model_output_token_clear'])#change block or token or line

    temp_score_1 = compute_score_k(answer, model_output_list, 1)
    temp_score_3 = compute_score_k(answer, model_output_list, 3)
    temp_score_10 = compute_score_k(answer, model_output_list, 10)

    score_list_1.append(temp_score_1)
    score_list_3.append(temp_score_3)
    score_list_10.append(temp_score_10)



pass_at_1 = sum(score_list_1)/len(score_list_1)
pass_at_3 = sum(score_list_3)/len(score_list_3)
pass_at_10 = sum(score_list_10)/len(score_list_10)

pass_at = {
    1: pass_at_1,
    3: pass_at_3,
    10: pass_at_10,
}

for k in ks:
    print(f"pass@{k} = {pass_at[k]}")


