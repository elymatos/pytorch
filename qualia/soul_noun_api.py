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
Given your profound knowledge about words, lexical items and concepts, classify the following Brazilian Portuguese nouns according to these semantic classes (each class contains some exemplary words):

@telic: target, objective, purpose, goal.
@agentive: cause, origin, provenance, reason, motive.
@cause: cause, get, induce, make, create, cause something to happen, produce, avoid.
@part: Parts of entities belonging to simple parts. piece, bit, portion, leaf, peel, bark, stone, brush.
@body_part: Body parts of humans and animals. hand, head, eye, paw, tail, muzzle.
@group: group, set, drove, flight, flock, pack, shoal, collection, pair.
@human_group: Collective nouns which are composed by humans. band, senate, audiency, home.
@amount: drop, spoon, bit, pinch , heap, quantity, degree, pile, myriad.
@entity: Nouns denoting stable things in world. god, spirit, thing.
@locale: Geographical locations or to natural locations. place, location.
@geopolitical_location: Proper and common names referring to geopolitical locations.. nation, city, district, village.
@natural_feature: Natural locations which are areas, surfaces or 3D locales. beach, clearing, field, cave, mountain.
@opening: hole, tunnel, window, door.
@building: bank, school, house, shack, igloo.
@artifactual_area: Artifactual locations which are areas or surfaces. square, road.
@material: Entities of different types used to produce something else.
@artifact: Substances, locations, objects, etc. produced by man with a purpose.
@artifactual_material: Artifactual substances or objects used as materials to produce something else. concrete, plastic, leather, brick, wire, fur.
@furniture: table, chair, closet, cupboard.
@utensils:
@toys:
@clothing: Dress, jacket, shirt, trousers, shoes, handbag.
@container: container, box, bottle, glass.
@physical_artwork: painting, sculpture, drawing.
@instrument: Tools of different sorts used to perform some task.
@money: objects that have a financial value.
@vehicle: Entities used for transportation, and in general to move. vehicle, car, ship.
@semiotic_artifact: Physical objects supporting information.book, document, card, letter.
@food: Entities that are used for alimentation.
@artifact_food: Food prepared in order to be eaten.
@ingredients: Condiment, spice, pepper, salt, parsley, vinegar, dressing.
@physical_object: Shaped, natural, non-living entities. stone, rock.
@organic_object:natural entities sometimes produced by other living entities. Egg, body.
@animal:
@human:
@people: Nouns related to people in genral.
@people_by_age:individuals as viewed in terms of their age.
@people_by_religion:individuals with respect to their Religion.
@people_by_jurisdiction:individuals who are governed by virtue of being registered in a certain Jurisdiction.
@people_by_morality: individuals whose specific acts or general behavior is evaluated against standards of morality or rightness.
@people_along_political_spectrum: An Individual in a group categorized in terms of the political views.
@People_by_leisure_activity: people considered in regards to some leisure activity in which she is engaged.
@people_by_origin:individuals with respect to their Origin.
@people_by_ethnicity: humans with respect to their Ethnicity.
@people_by_transitory_activity:tenant, student, murderer, pedestrian, passenger, patient, fugitive, neighbourgh.
@people_by_gender_or_sexual_orientation:
@people_by_health_condition:
@people_with_disabilities:
@people_by_persistent_activity:violinist, sailor.
@people_by_profession: lawyer, butcher, waiter, major-general, journalist.
@role: someone is a member or part of a group. member, follower, adherent.
@people_by_ideo: communist, Christian, Jewish, freudian. people which follow some ideological movement.
@kinship: mother, father, brother, son.
@social_status: lord, leader, etc. people having a special social role in different fields: religion, aristocracy, government.
@vegetal:
@fruit:
@micro-organism: bacterium, virus, staphylococcus, plancton.
@substance: unshaped entities, natural substances, chemical substances. substance, matter, stuff.
@natural_substance: Uranium, silver, sweat, clay, wood, sand, marble etc.
@substance_food: honey, lamb, chicken, meat, etc.
@drink: drink, beverage. substances that are used for drinking.
@artifactual_drink: Cocktail, wine, beer. artifactual substances that are used for drinking.
@quality: beauty, goodness, ugliness, courage, dirtiness. Deadjectival nouns.
@psychological_prperty:faculty, intelligence, courage, confidence, attention, care, intuition, reason, thought, cynism.
@dimension:colour, shape, measurement, magnitude, dimension, value, lenght, size, weight, temperature, speed.
@sense: nouns related to senses. smell, taste, hearing, sight, touch, etc.
@social_property:power, authority, right.
@domain: area of knowledge or activity. discipline, field, medicine, physics, breeding, journalism, trade, agriculture, research, teaching.
@time: nouns referring to temporal expressions: time, point of time, periods of time, or parts of processes or events.
@moral_standards: Right, good, perfection, charity, equality, fraternity, freedom. This template encodes moral principles which affect people's attitudes and behaviour.
@cognitive_fact:knowledge, thought, theory, notion, interpretation.
@movement_of_thought:Communism, Marxism, Socialism, Romanticism.
@institution:Bank, museum, school, church, party, etc. This class contains names referring to human/social institutions and organizations.
@convention:law, tax, bill, norm, regulations, agreement, contract. Convention used in society and culture.
@representation:representation, symbol, sign. Nouns the represents something.
@language: the different languages.
@sign:Word, point, comma, brackets, sentence, letter, icon. Words are used to convey information and  meaning.
@information:book, letter, document, message. Nouns that denote the information contained in a support of information.
@number:ordinal or cardinal numbers.
@unit_of_measurement:Meter, mile, kilo, litre, degree, foot, hour, pound, year, month, day.
@business:

Please, use preferably the given classes but you can extend for your inferred classes.

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