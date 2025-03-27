
from flask import Flask, request, jsonify

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
