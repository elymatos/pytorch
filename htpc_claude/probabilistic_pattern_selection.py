def probabilistic_pattern_selection(candidates, bigram_memory, phrase_memory):
    """
    Select a token from candidates based on frequency counts in the model.

    Args:
        candidates: List of candidate tokens
        bigram_memory: Dictionary of bigram frequencies
        phrase_memory: Dictionary of phrase frequencies

    Returns:
        Selected token and its probability score
    """
    if not candidates:
        return None, 0.0

    scores = {}
    total_score = 0

    for token in candidates:
        # Start with a base score
        score = 1.0

        # Consider previous context (last few tokens)
        context = [tok for sublist in current_pattern[-3:] for tok in sublist if sublist]

        # Add bigram influence
        for prev_token in context:
            bigram = (prev_token, token)
            if bigram in bigram_memory:
                score *= (1 + bigram_memory[bigram])

        # Add phrase influence
        flat_context = [tok[0] for tok in current_pattern[-3:] if tok]
        for phrase in phrase_memory:
            flat_phrase = [phrase[0][0]] + [pair[1] for pair in phrase]
            for i in range(min(len(flat_context), len(flat_phrase))):
                if flat_context[-i:] == flat_phrase[:i] and i < len(flat_phrase) - 1 and flat_phrase[i] == token:
                    score *= (1 + phrase_memory[phrase])

        scores[token] = score
        total_score += score

    # Normalize scores
    if total_score > 0:
        for token in scores:
            scores[token] /= total_score

    # Select based on probability
    if random.random() < 0.9:  # 90% follow probabilities, 10% explore
        selected = max(scores.keys(), key=lambda x: scores[x])
    else:
        selected = random.choice(list(scores.keys()))

    return selected, scores[selected]


# Then modify the recognition function to use this:
def recognize_patterns_probabilistic(token_matrix, token_transitions, bigram_memory,
                                     phrase_memory, phrase_hierarchy, pos_dict,
                                     noise_probability=0.0):
    # ... existing code ...

    for i in range(len(token_matrix)):
        curr_tokens = token_matrix[i]
        prev_tokens = token_matrix[i - 1] if i > 0 else [None]

        print(f"🔎 Posição {i}: opções = {curr_tokens}")

        # Filter valid candidates
        valid_candidates = []
        for curr_token in curr_tokens:
            for prev_token in prev_tokens:
                is_valid_transition = (
                        token_transitions.get(prev_token) == curr_token or
                        (prev_token, curr_token) in bigram_memory
                )
                context = [tok for sublist in current_pattern[-3:] for tok in sublist]
                top_down = dynamic_feedback_for_context(context, phrase_memory, phrase_hierarchy, noise_probability)
                top_down_match = curr_token in top_down or not top_down

                if is_valid_transition and top_down_match:
                    valid_candidates.append(curr_token)

        # Select probabilistically
        selected_token, score = probabilistic_pattern_selection(valid_candidates, bigram_memory, phrase_memory)

        if selected_token:
            print(f"  → Selecionado: {selected_token} (score: {score:.4f})")
            current_pattern.append([selected_token])
            matched = True
        else:
            matched = False

        # ... rest of the function ...