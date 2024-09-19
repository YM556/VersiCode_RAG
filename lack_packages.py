import os
import json
from tqdm import tqdm
corpus_dir = "/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/library_source_code/version_corpus"

corpus_list = os.listdir(corpus_dir)

sources = ["library_source_code","downstream_application_code","stackoverflow"]
# sources = ["library_source_code"]

lack_packages = []
lack_versions = {}
cnt1 = 0
cnt2=0
all_data=0

for sample_source in sources:
    json_path = os.path.join('/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/',sample_source,'{}_token.json'.format(sample_source))
    with open(json_path,"r",encoding="utf-8") as f:
        lodic = json.load(f)
        data_list = lodic["data"]
        for item in tqdm(data_list,total=len(data_list)):
            all_data += len(data_list)
            dep = item["dependency"]
            if dep not in corpus_list:
                cnt1+=1
                if dep not in lack_packages:
                    lack_packages.append(dep)
                with open ("./lack_package.json","w") as pf:
                    json.dump(lack_packages,pf,ensure_ascii=False, indent=4)
            else:
                _version_list = os.listdir(os.path.join(corpus_dir,dep))
                version_list = [version[:-6] for version in _version_list]
                version = item["version"][2:]
                print(version)

                if version not in version_list:
                    cnt2+=1
                    if dep not in lack_versions.keys():
                        lack_versions[dep] = []
                    if version not in lack_versions[dep]:
                        lack_versions[dep].append(version)
                    with open("./lack_version.json","w") as vf:
                        json.dump(lack_versions,vf,ensure_ascii=False, indent=4)
                        json.dump("\n",vf)


print("Total number: {}".format(cnt1+cnt2))
print("lack of package: {}".format(cnt1))
print("lack of matching version: {}".format(cnt2))
print("All sample numbers: {}".format(all_data))
