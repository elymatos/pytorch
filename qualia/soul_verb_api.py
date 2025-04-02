import os
from flask import Flask, request, jsonify

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

system_message = """
Given your profound knowledge about words, lexical items and concepts, classify the following Brazilian Portuguese verbs according to these semantic classes:

@causative_act: an intentional action performed by an agent
@speech_act: an act relative to linguistic or verbal expressions (talk, speak)
@movement: represents some kind of movement or change of location
@causative_movement: a movement caused by some force or event
@non_relational_act: an intentional action performed by an agent that has no effect in a patient. The verb requires just one argument.
@relational_act: an intentional action performed by an agent but that affects a patient. The verb requires at least two arguments.     
@aspectual: aspectual event (e.g., begin, stop) in a non causative reading
@cause_aspectual: aspectual event (e.g., begin, stop) in a causative reading  
@causative: an event (not an agent) causes a change in world, situation, or entity
@causative_relational: an event causes a change of state, of value or a constituive change in a patient. The verb requires at least two arguments.  
@change: indicates a change or transition (in location, possession, state, constitution, etc), regardless of cause  
@natural_transition: a change or transition the occurs naturally
@creation: a change that produces a new thing
@stative: state or condition of an entity that tends to remains along the time  
@existence: verbs related to the existince of something
@relational_state: stative situations the envolves more than one participant (location, possession, constitution, etc)
@phenomenon: non-agentive subject (e.g., natural phenomenon, stimulus, disease)  
@psy_event: psychological events
@cognitive_event: events related to cognition and thinking
@emotion: events related to emotions or feeling
@perception: events related to perception
@experience: events in which an experiencer has some perception
@modal: verbs related to modality (epistemic, deontic, etc)    

Please format the response associating a word with a list of classes using a JSON-like format.

\n{format_instructions}\n
Verbs: {query}\n
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
prompt = PromptTemplate(
    template=system_message,
    input_variables=["query", "relations"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

# messages = [
#    SystemMessage(system_message),
#    HumanMessage("cantar, correr, comprar, adoecer, chove, estar, permanecer, ser, terminar"),
# ]
#
# response = llm.invoke(messages)
# print(response)

app = Flask(__name__)


@app.route('/soul', methods=['POST'])
def predict_classes():
    print(request);
    data = request.get_json()
    print(data)
    words = data.get("words")
    # print(words)
    # generator_expr = (str(element) for element in words)
    # separator = ','
    # result_string = separator.join(generator_expr)
    # print(result_string)

    response = chain.invoke({"query": words})

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