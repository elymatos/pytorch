import os
from flask import Flask, request, jsonify

os.environ[
    "OPENAI_API_KEY"] = ""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

system_message = """
Given your profound knowledge about words, lexical items and concepts, I'll give you a list of templates of sentences that represents a relation between two words ans a list of words. 
In each template sentence in the list the expression %w1 represents a given word and the expression %w2 represents a related word.
For each word in the given list, you must process all the sentence templates, replacing the %w2 expression with a related word.
Note, please, that not all the given words will use all sentences. Give me just those that make sense from a human point of view.
Don't use metaphorical or mythological relations or interpretations. Use your linguistic (syntactic and grammatical) and conceptual knowledge to complete the sentence.
Here is the list of templates, separated by commas: {relations}.
If possible, structure your response in a JSON-like format. In the response, please make just the word for expression %w2 explicit, not the whole template text.
Try to get at least 5 words for each expression %w2 in a template.
The given words must be considered nouns (entities, objects, artifacts, material, etc).
 The words are written in Brazilian Portuguese, so the related words must be also writen in Brazilian Portuguese.
\n{format_instructions}\n{query}\n
"""

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

parser = JsonOutputParser()
relations = [
    '%w1 causes %w2 in a agentive way',
    '%w1 uses %w2 in a agentive way',
    '%w1 causes %w2 indirectly',
    'the occurrence of %w1 can cause emotion %w2',
    '%w2 is caused by %w1 by natural means',
    'the occurrence of %w1 can cause the reaction %w2',
    'the occurrence of %w1 can cause the infection %w2',
    '%w1 uses %w2 in a constitutive way',
    '%w1 contains %w2',
    '%w1 is caused by the agent %w2',
    '%w1 is experienced by %w2',
    'the occurrence of %w1 can cause the emotion %w2',
    '%w1 has the attribute %w2',
    '%w2 is a typical syntactic subject of %w1',
    '%w2 is a typical syntactic object of %w1',
    '%w2 is a typical semantic agent of %w1',
    '%w2 is a typical semantic patient of %w1',
    '%w1 typically occurs in the context of %w2',
    '%w2 typically occurs in the context of %w1',
    '%w1 has %w2 as source or origin',
    '%w1 is the effect of %w2',
    '%w1 is an instrument for %w2',
    '%w1 is member of %w2',
    '%w1 is the source or origin of %w2',
    '%w1 is the typical location of %w2',
    '%w1 influences %w2',
    '%w1 includes %w2',
    '%w1 is a follower of %w2',
    '%w1 is a instance or example of %w2',
    '%w1 is a kind of %w2',
    '%w1 is a symbol of %w2',
    '%w1 is the vice of %w2',
    '%w1 is the activity of %w2',
    '%w1 is affected by %w2',
    '%w1 is associated to %w2',
    '%w1 is caused intentionally by %w2',
    '%w1 is created by %w2',
    '%w1 derives from %w2',
    '%w1 is found in %w2',
    '%w1 is made of %w2',
    '%w1 is measured by %w2',
    '%w1 is part of %w2',
    '%w1 is regulated by %w2',
    '%w1 is resolved by %w2',
    '%w1 is a subtype of %w2',
    '%w1 is the ability of %w2',
    '%w1 is the activity performed in %w2',
    '%w1 is the habit of %w2',
    '%w1 is the purpose of %w2',
    '%w1 is the resulting state of %w2',
    '%w1 is the workplace of %w2',
    '%w1 is typical of %w2',
    '%w1 is used against %w2',
    '%w1 is used as %w2',
    '%w1 is used by %w2',
    '%w1 is used during %w2',
    '%w1 is used for %w2',
    '%w1 is valued by %w2',
    '%w1 is a kinship to %w2',
    '%w1 lives in %w2',
    '%w1 measures %w2',
    '%w1 precedes %w2',
    '%w1 succeeds %w2',
    '%w1 produces %w2 naturally',
    '%w1 quantifies %w2',
    '%w1 results from %w2',
    '%w1 results in %w2',
    '%w1 is located at %w2',
    '%w1 is related to %w2',
    '%w2 is a resulting state of %w1',
    '%w1 is a inherent activity of %w2',
    '%w1 occurs while %w2',
]

generator_expr = (str(element) for element in relations)
separator = ','
relations_string = separator.join(generator_expr)

prompt = PromptTemplate(
    template=system_message,
    input_variables=["query", "relations"],
    partial_variables={"format_instructions": parser.get_format_instructions(), "relations": relations_string},
)

chain = prompt | llm | parser

# messages = [
#    SystemMessage(system_message),
#    HumanMessage("quebrar"),
# ]

# response = llm.invoke(messages)
# print(response)

app = Flask(__name__)


@app.route('/relations', methods=['POST'])
def predict_next_concepts():
    data = request.get_json()
    words = data.get("words")
    generator_expr = (str(element) for element in words)
    separator = ','
    result_string = separator.join(generator_expr)
    response = chain.invoke({"query": result_string})
    print(response)
    return response


if __name__ == '__main__':
    app.run(debug=True, port=5001)
