import os
from flask import Flask, request, jsonify

os.environ["OPENAI_API_KEY"] = ""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

system_message = """
Given your profound knowledge about word (lexical items), I'll give you a list of relations and a word. 
Please, for each relation you generate a list of words (lexical items) related to the given word. Generate just relation that make sense
from a human point of view. Don't use metaphorical or mythological relations.
This is the list of relations, separated by commas: is affected agentively by,agentivally causes,agentivally uses,has as agentive cause,has as agentive experience,is associated with,is caused agentively by,is caused by,is caused intentionally by,causes emotion,causes infection,causes naturally,causes reaction,causes,constitutively uses,contains,is created by,is derived from,is found in,has as attribute,has as building part,has as colour,has as effect,has as instrument,has as member,has as origin,has as part,has as source,has as typical location,has diagnostic by,has influence on,is idiosincrasy of,includes,is a instance of,is a follower of,is a kind of,is activity of,is made of,is regulated by,is the ability of,is the activity of,is the activity performed in,is the habit of,is the purpose of,is the workplace of,kinship,lives in,made of,is measured by,measures,is part of,precedes,produces naturally,quantifies,is generally related to,is resolved by,results from,is the result state of,results in,is subtype of,is a symbol of,is typical of,is used against,is used as,is used by,is used during,is used for,is valued by,is a vice of.
Note, please, that not all word will use all relations. Give me just those that make sense from a human point of view.
If possible, structure your response in a JSON-like format.The given word is written in Brazilian Portugues, so the related words must be also writen in Brazian Portuguese.
\n{format_instructions}\n{query}\n
"""

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template=system_message,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser


#messages = [
#    SystemMessage(system_message),
#    HumanMessage("quebrar"),
#]

#response = llm.invoke(messages)
# print(response)

app = Flask(__name__)

@app.route('/relations', methods=['POST'])
def predict_next_concepts():
    print(request);
    data = request.get_json()
    print(data)
    words = data.get("words")
    print(words)
    generator_expr = (str(element) for element in words)
    separator = ','
    result_string = separator.join(generator_expr)
    print(result_string)

    response = chain.invoke({"query": result_string})

    # messages = [
    #     SystemMessage(system_message),
    #     HumanMessage(result_string),
    # ]
    #
    # response = llm.invoke(messages)
    print(response)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001)
