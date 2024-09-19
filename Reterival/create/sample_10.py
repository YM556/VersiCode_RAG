import json
import os.path

names = ["downstream_application_code"]
for name in names:
    data_output = []
    dependency_list = []
    index=0
    input_path = os.path.join("/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/",name,name+"_token.json")
    output_path = os.path.join(
        "/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/", name,
        "samples", "sample_10.jsonl")
    with open(input_path,"r",encoding="utf-8") as inf:
        lodict = json.load(inf)
        data_list = lodict["data"]
        for item in data_list:
            dependency = item["dependency"]
            # version = item["version"][2:]
            while index<10:
                corpus_dir = os.path.join("/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/",name,"version_corpus",dependency+".jsonl")
                if os.path.exists(corpus_dir) and dependency not in dependency_list:
                    dependency_list.append(dependency)
                    print("add")
                    with open(output_path, "a") as outf:
                        outf.write(json.dumps(item)+'\n')
                    index += 1
                else:
                    break
    print("{} is done".format(name))

