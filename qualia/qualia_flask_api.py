
from flask import Flask, request, jsonify

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
#%%
api_key = "tgp_v1_FnWpQqvXdjUBRN3MVUSvzk8Ny4EZ7kcZiDMWQM3KUA4"

#json_structure = """{
#  "word": "book",
#  "relation": "is made of",
#  "items": [
#      "paper"
#  ]}"""

word = """book"""

system_message = """
Given your profound knowledge about word (lexical items), I'll give you a list of relations and a word. 
Please, for each relation you generate a list of words (lexical items) related to the given word. Generate just relation that make sense
from a human point of view. Don't use metaphorical or mythological relations.
This is the list of relations, separated by commas: is affected agentively by,agentivally causes,agentivally uses,has as agentive cause,has as agentive experience,is associated with,is caused agentively by,is caused by,is caused intentionally by,causes emotion,causes infection,causes naturally,causes reaction,causes,constitutively uses,contains,is created by,is derived from,is found in,has as attribute,has as building part,has as colour,has as effect,has as instrument,has as member,has as origin,has as part,has as source,has as typical location,has diagnostic by,has influence on,is idiosincrasy of,includes,is a instance of,is a follower of,is a kind of,is activity of,is made of,is regulated by,is the ability of,is the activity of,is the activity performed in,is the habit of,is the purpose of,is the workplace of,kinship,lives in,made of,is measured by,measures,is part of,precedes,produces naturally,quantifies,is generally related to,is resolved by,results from,is the result state of,results in,is subtype of,is a symbol of,is typical of,is used against,is used as,is used by,is used during,is used for,is valued by,is a vice of.
Note, please, that not all word will use all relations. Give me just those that make sense from a human point of view.
If possible, structure your response in a JSON-like format.
"""

llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_tokens=1024,
    )

#    structured_llm = llm.with_structured_output(json_structure, method="json_mode")
#    system_prompt = system_message.format(json_structure=json_structure)

messages = [
    SystemMessage(system_message),
    HumanMessage("book"),
]

response = llm.invoke(messages)
print(response.content)

app = Flask(__name__)

# Simulated semantic graph
semantic_graph = {
    "dog": {"chased": 0.33, "barked": 0.33, "ate": 0.33},
    "cat": {"climbed": 1.0},
    "chef": {"baked": 0.5, "sliced": 0.5},
    "baked": {"bread": 1.0},
    "sliced": {"bread": 1.0},
    "mailman": {"delivered": 1.0},
    "delivered": {"letter": 1.0}
}

@app.route('/predict', methods=['POST'])
def predict_next_concepts():
    data = request.get_json()
    print(data)
    current_word = data.get("current_word")
    top_k = int(data.get("top_k", 3))
    context_sequence = data.get("context_sequence", [])

    predictions = semantic_graph.get(current_word, {})
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_preds = [{"concept": w, "score": round(p, 2)} for w, p in sorted_preds[:top_k]]

    return jsonify({
        "semantic_predictions": top_preds,
        "source_word": current_word,
        "context": context_sequence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
