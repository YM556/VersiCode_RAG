import os.path

import argparse

from Reterival.create.utils import load_jsonlines,write_jsonl

# query : id->str,text->str
def document2code(data:list,from_version,to_version):
    data_queries = []
    for index,line in enumerate(data):

        from_v = line[from_version]
        to_v = line[to_version]
        id = line["id"]
        dependency = line["dependency"]
        text = line["description"]
        data_queries.append({"id":id,"dependency":dependency,"text":text+"from version{} to version {}".format(from_v,to_v)})

    return data_queries


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/Volumes/kjz-SSD/Datasets/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_editing/",
                        help="The root path of your dataset dir")
    parser.add_argument("--output_dir", type=str, default="../datasets/code_editing/new_to_old")
    parser.add_argument("--file_name",type=str,default="code_editing_new_to_old.json")
    args = parser.parse_args()

    if args.file_name=="code_editing_old_to_new.json":
        from_version = "old_version"
        to_version = "new_version"
    else:
        from_version = "new_version"
        to_version = "old_version"


    output_path = os.path.join(args.output_dir,r"queries.jsonl")

    file_path = os.path.join(args.root_dir,args.file_name)

    data_list = load_jsonlines(file_path)
    data_queries = document2code(data_list,from_version,to_version)
    write_jsonl(data_queries,output_path)


if __name__ == '__main__':
    main()