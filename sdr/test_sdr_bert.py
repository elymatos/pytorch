import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from htm.bindings.sdr import SDR

# Configuration
BERT_MODEL = "bert-base-uncased"
TOP_K = 20
SDR_DIM = 768

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
model = BertModel.from_pretrained(BERT_MODEL)
model.eval()


def text_to_sdr(text: str, top_k=TOP_K) -> SDR:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs.last_hidden_state[0, 0, :].numpy()

    # Get top-k activations
    top_k_indices = np.argpartition(cls_vector, -top_k)[-top_k:]

    sdr = SDR([SDR_DIM])
    sdr.setSparseIndices(sorted(top_k_indices))
    return sdr


def similarity(sdr1: SDR, sdr2: SDR) -> float:
    overlap = len(set(sdr1.getSparse()) & set(sdr2.getSparse()))
    union = len(set(sdr1.getSparse()) | set(sdr2.getSparse()))
    return overlap / union if union != 0 else 0.0


# Example
text1 = "The scientist published a breakthrough paper."
text2 = "A researcher released an important article."

sdr1 = text_to_sdr(text1)
sdr2 = text_to_sdr(text2)

print("SDR 1 active bits:", sdr1.getSparse())
print("SDR 2 active bits:", sdr2.getSparse())
print("Overlap:", len(set(sdr1.getSparse()) & set(sdr2.getSparse())))
print("Jaccard similarity:", similarity(sdr1, sdr2))
