import json

from knowledge_graph_maker import GraphMaker, Ontology, GroqClient, OpenAIClient
from knowledge_graph_maker import Document

data_list = []

with open("/root/autodl-tmp/VersiCode/data/VersiCode_Benchmark/VersiCode_Benchmark/code_completion/library_source_code/torch_samples.jsonl",'r') as f:
    for line in f:
        item = json.loads(line)
        data_list.append(str(item))

data_text = ''.join(data_list)

ontology = Ontology(
    # labels of the entities to be extracted. Can be a string or an object, like the following.
    labels=[
        {"dependency": "The dependency name of python"},
        {"version": "The version number corresponding to the dependency"},
        {"API":"API names from this dependency"},
        {"description": "The text which used to describe the using of the API"},
    ],
    # Relationships that are important for your application.
    # These are more like instructions for the LLM to nudge it to focus on specific relationships.
    # There is no guarentee that only these relationships will be extracted, but some models do a good job overall at sticking to these relations.
    relationships=[
        "Relation between any pair of Entities",
        "Relation between APIs with different version number"
        ],
)
#  export GROQ_API_KEY="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"
model = "gemma-7b-it"
llm = GroqClient(model=model, temperature=0.1, top_p=0.5)


import datetime
current_time = str(datetime.datetime.now())
def generate_summary(text):
    SYS_PROMPT = (
        "Succintly summarise the text provided by the user. "
        "Respond only with the summary and no other comments"
    )
    try:
        summary = llm.generate(user_message=text, system_message=SYS_PROMPT)
    except:
        summary = ""
    finally:
        return summary


docs = map(
    lambda t: Document(text=t, metadata={"summary": generate_summary(t), 'generated_at': current_time}),
    data_text
)

graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=False)
graph = graph_maker.from_documents(
    list(docs),
    delay_s_between=10 ## delay_s_between because otherwise groq api maxes out pretty fast.
    )
print("Total number of Edges", len(graph))