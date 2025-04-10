import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse import csr_matrix

# Configuration
MODEL_NAME = "bert-base-uncased"
TOP_K = 20
SDR_DIM = 768

# Load BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()

def bert_to_sparse_vector(text: str, top_k: int = TOP_K) -> csr_matrix:
    """Encodes text using BERT and sparsifies using top-k elements, returns sparse CSR vector."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs.last_hidden_state[0, 0, :].numpy()

    # Keep top-k indices only
    top_k_indices = np.argpartition(cls_vector, -top_k)[-top_k:]
    sparse_data = np.zeros_like(cls_vector)
    sparse_data[top_k_indices] = cls_vector[top_k_indices]

    # Convert to CSR matrix (1 row, SDR_DIM columns)
    return csr_matrix(sparse_data.reshape(1, -1))
# Example sentences
text_a = "The cat chased the mouse."
text_b = "A dog ran after a rodent."

vec_a = bert_to_sparse_vector(text_a)
vec_b = bert_to_sparse_vector(text_b)

# Stack them into one sparse matrix
matrix = csr_matrix(np.vstack([vec_a.toarray(), vec_b.toarray()]))

# Compute top-1 cosine similarity using sparse-dot-topn
similarity_matrix = awesome_cossim_topn(matrix, matrix, ntop=1, lower_bound=0.0, use_threads=True, n_jobs=1)

similarity_score = similarity_matrix[0, 1]

print("Cosine similarity (sparse):", similarity_score)
