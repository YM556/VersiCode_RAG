import os.path

import argparse

from Reterival.create.utils import load_jsonlines,write_jsonl

# query : id->str,text->str
def document2code(data:list):
    data_queries = []
    for index,line in enumerate(data):
        id = line["id"]
        version = line["version"]
        dependency = line["dependency"]
        text = line["description"]
        data_queries.append({"id":id,"dependency":dependency+version,"text":text})

    return data_queries


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/",
                        help="The root path of your dataset dir")
    parser.add_argument("--output_dir", type=str, default="../datasets")
    parser.add_argument("--granularity",type=str,help="The query from different level. including block, line,token",default="token")
    parser.add_argument("--task", type=str,help="library_source_code,downstream_application_code, stackoverflow",
                        default="library_source_code")
    parser.add_argument("--language",help="If task is stackoverflow, language can be set as python java javascript c#")


    args = parser.parse_args()

    if args.task=="stackoverflow":
        if args.language == "python":
            file_name = r"stackoverflow_{}.json".format(args.granularity)
            output_path = os.path.join(args.output_dir, args.task,args.language,args.granularity, r"queries.jsonl")
            file_path = os.path.join(args.root_dir,args.task, args.language, file_name)
        else:
            file_name = r"{}_block.json".format(args.language)
            output_path = os.path.join(args.output_dir, args.task, args.language, r"queries.jsonl")
            file_path = os.path.join(args.root_dir, args.task, args.language, file_name)

    else:

        file_name = args.task + '_' + args.granularity + '.json'
        output_path = os.path.join(args.output_dir, args.task,r"queries_{}.jsonl".format(args.granularity))
        file_path = os.path.join(args.root_dir, args.task,file_name)

    data_list = load_jsonlines(file_path)
    data_queries = document2code(data_list)
    write_jsonl(data_queries,output_path)


if __name__ == '__main__':
    main()