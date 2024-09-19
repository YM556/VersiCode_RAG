import argparse
import json
import os
import re
import jsonlines


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def load_jsonlines(file):
    json_data = []
    with open(file,'r') as f:
        for line in f:
            json_data.append(json.loads(line.strip()))
    return json_data

def document2code(data,fileName):
    data_corpus = []
    if fileName[:-6]== "downstream_application_code_raw_data":
        for idx, item in enumerate(data):
            version_pattern = re.compile(r'<version>(.*?)</version>\n')
            cleaned_text = re.sub(r'<version>(.*?)</version>|<filePath>(.*?)</filePath>','',item['data'])
            id = fileName+str(idx)
            version = version_pattern.findall(item["data"])[0]
            text = cleaned_text
            data_corpus.append({"id":id,"version":version,"text":text,})
    elif fileName[:-6]== "library_source_code_raw_data3" or fileName[:-6]== "library_source_code_raw_data":
        for idx, item in enumerate(data):
            version_pattern = re.compile(r'<version>(.*?)</version>')
            filePath_pattern = re.compile(r'<filePath>(.*?)</filePath>')

            cleaned_text = re.sub(r'<version>(.*?)</version>|<filePath>(.*?)</filePath>', '', item['data'])
            id = fileName + str(idx)
            title = ''.join(version_pattern.findall(item["data"])) + ''.join(filePath_pattern.findall(item["data"]))

            text = cleaned_text
            data_corpus.append({"id": id, "title": title, "text": text})
    elif fileName[:-6]=="stackoverflow_raw_data":
        for idx,item in enumerate(data):
            question_body = item["question_body"]
            answer_body = item["answer_body"]
            id = fileName+str(idx)
            text = question_body+answer_body
            data_corpus.append({"id": id, "text": text})
    else:
        for idx, item in enumerate(data):
            title = item["title"]
            id = fileName+str(idx)
            text = item["question_body"]+item["answer_body"]

            data_corpus.append({"id":id,"title":title,"text":text,"metadata":{}})



    return data_corpus

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",type=str,default="/Volumes/kjz-SSD/Datasets/VersiCode_Raw/VersiCode_Raw/")
    parser.add_argument("--output_dir", type=str, default="../datasets")
    parser.add_argument("--file_name", type=str, default="downstream_application_code_raw_data.jsonl")
    parser.add_argument("--task", type=str, help="library_source_code,downstream_application_code, stackoverflow",
                        default="downstream_application_code")
    args = parser.parse_args()

    file = os.path.join(args.root_dir,args.file_name)

    file_path = os.path.join(args.root_dir,file)
    dataset = load_jsonlines(file_path)
    corpus = document2code(dataset,args.file_name)

    path = args.output_dir
    save_file_jsonl(corpus,os.path.join(path,args.task,"corpus.jsonl"))

if __name__ == "__main__":
    main()



