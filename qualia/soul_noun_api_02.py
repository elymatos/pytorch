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
class ClassifiedNoun(BaseModel):
    palavra: str
    classes: List[str]
    definition: Optional[str] = Field(None)
    examples: Optional[List[str]] = Field(None)

class ClassificationResult(RootModel[List[ClassifiedNoun]]):
    pass

# -------------------------------
# File Paths and Parameters
# -------------------------------
ontology_path = "noun_ontology_fully_described.json"
input_txt_path = "nouns.txt"
output_json_path = "resultados_classificacao_com_expansao.json"

BATCH_SIZE = 100

# -------------------------------
# Load Ontology and Input Words
# -------------------------------
with open(ontology_path, "r", encoding="utf-8") as f:
    ontology_json_content = f.read()

with open(input_txt_path, "r", encoding="utf-8") as f:
    palavras = [line.strip() for line in f if line.strip()]

# -------------------------------
# Prompt Template and LLM Setup
# -------------------------------
parser = PydanticOutputParser(pydantic_object=ClassificationResult)

prompt_template = PromptTemplate(
    input_variables=["nouns"],
    partial_variables={
        "ontology": ontology_json_content,
        "format_instructions": parser.get_format_instructions()
    },
    template="""
You are a lexical semantics expert.

Below is an ontology of semantic categories for nouns. Each category includes a name (prefixed with "@"), a definition, and example words.

Your task is to classify the following Brazilian Portuguese nouns using the most appropriate categories from this ontology.

If a noun clearly fits into one or more categories, use them directly.

However, if no category in the ontology applies semantically, you may invent a new category. In that case:
- Prefix it with "@NEW_"
- Provide a short definition
- Provide 1–3 example nouns (besides the one being classified)

Return the result as a JSON array. Each entry must include:
- "palavra": the noun being classified
- "classes": list of semantic categories (existing or new)
- If any class starts with "@NEW_", also include:
  - "definition": brief description of the new class
  - "examples": example words (in addition to the one classified)

Ontology (in JSON format):
{ontology}

List of nouns to classify:
{nouns}

{format_instructions}
""",
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# -------------------------------
# Function to Classify One Batch
# -------------------------------
def classify_nouns_with_expansion(nouns):
    formatted_prompt = prompt_template.format_prompt(nouns=json.dumps(nouns, ensure_ascii=False))
    messages = formatted_prompt.to_messages()
    response = llm(messages)
    return parser.parse(response.content).root

# -------------------------------
# Run Classification in Batches
# -------------------------------
all_results = []
for i in tqdm(range(0, len(palavras), BATCH_SIZE), desc="Classificando batches"):
    batch = palavras[i:i + BATCH_SIZE]
    try:
        result = classify_nouns_with_expansion(batch)
        all_results.extend(result)
    except Exception as e:
        print(f"Erro no batch {i // BATCH_SIZE + 1}: {e}")

# -------------------------------
# Save Output as JSON
# -------------------------------
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump([r.dict() for r in all_results], f, ensure_ascii=False, indent=2)

print(f"\n✅ Classificação finalizada! Resultado salvo em: {output_json_path}")
