import os

from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain
#from langchain_core.output_parsers import JsonOutputParser
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
import json

# ✅ NEW (for Pydantic v2)
from pydantic import RootModel

# -------------------------------
# Pydantic Schema for Output
# -------------------------------
class ClassifiedVerb(BaseModel):
    verbo: str
    classes: List[str]
    definition: Optional[str] = Field(None)
    examples: Optional[List[str]] = Field(None)

class VerbClassificationResult(RootModel[List[ClassifiedVerb]]):
    pass

ontology_path = "verb_ontology_with_definitions.json"
input_txt_path = "lista_de_verbos.txt"
output_json_path = "resultados_classificacao_verbos.json"

BATCH_SIZE = 100

with open(ontology_path, "r", encoding="utf-8") as f:
    ontology_json_content = f.read()

with open(input_txt_path, "r", encoding="utf-8") as f:
    verbos = [line.strip() for line in f if line.strip()]

parser = PydanticOutputParser(pydantic_object=VerbClassificationResult)

prompt_template = PromptTemplate(
    input_variables=["verbs"],
    partial_variables={
        "ontology": ontology_json_content,
        "format_instructions": parser.get_format_instructions()
    },
    template="""
Você é um especialista em semântica lexical.

Abaixo está uma ontologia de categorias semânticas para verbos. Cada categoria possui um nome (prefixado com "@"), uma definição e exemplos típicos de uso.

Sua tarefa é classificar os seguintes verbos em português brasileiro de acordo com a(s) categoria(s) mais apropriada(s) dessa ontologia.

Se um verbo se encaixa claramente em uma ou mais categorias, utilize-as diretamente.

Se nenhuma das categorias da ontologia for apropriada, você pode criar uma nova categoria. Nesse caso:
- Prefixe com "@NEW_"
- Forneça uma definição curta
- Dê 1 a 3 exemplos de outros verbos (além do verbo classificado)

Retorne o resultado como um array JSON. Cada item deve conter:
- "verbo": o verbo analisado
- "classes": uma lista com as categorias atribuídas
- Se alguma categoria começar com "@NEW_", inclua também:
  - "definition": uma breve definição da nova classe
  - "examples": verbos adicionais que pertenceriam a essa nova classe

Ontologia (em formato JSON):
{ontology}

Lista de verbos a classificar:
{verbs}

{format_instructions}
"""
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

def classify_verbs_with_expansion(verbs):
    formatted_prompt = prompt_template.format_prompt(verbs=json.dumps(verbs, ensure_ascii=False))
    messages = formatted_prompt.to_messages()
    response = llm(messages)
    return parser.parse(response.content).root

all_results = []
for i in tqdm(range(0, len(verbos), BATCH_SIZE), desc="Classificando verbos em lote"):
    batch = verbos[i:i + BATCH_SIZE]
    try:
        result = classify_verbs_with_expansion(batch)
        all_results.extend(result)
    except Exception as e:
        print(f"Erro no lote {i // BATCH_SIZE + 1}: {e}")

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump([r.dict() for r in all_results], f, ensure_ascii=False, indent=2)

print(f"\n✅ Classificação finalizada. Resultado salvo em: {output_json_path}")
