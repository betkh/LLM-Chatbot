import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROMPT_DIR = "prompts/"  # Directory where prompt files are stored


"""

GOAL : read prompt patterns from files and use them

file name same as prompt key

"""


def load_prompts():
    """
    Loads all prompt patterns from JSON files in the `prompts/` directory.
    """
    prompts = {}
    for filename in os.listdir(PROMPT_DIR):
        if filename.endswith(".json"):  # Load only JSON files
            key = filename.replace(".json", "")  # Extract key from filename
            with open(os.path.join(PROMPT_DIR, filename), "r") as file:
                # Read only the prompt text
                prompts[key] = json.load(file)["prompt"]
    return prompts


def select_best_prompt(user_query):
    """
    Finds the most relevant prompt pattern based on cosine similarity.
    """
    query_embedding = embedding_model.encode([user_query])
    similarity_scores = cosine_similarity(
        query_embedding, pattern_embeddings)[0]
    best_match_idx = similarity_scores.argmax()
    return pattern_texts[best_match_idx]  # Return best-matching prompt


PROMPT_PATTERNS = load_prompts()
print("PROMPT_PATTERNS =>>", PROMPT_PATTERNS)

# Recompute embeddings for dynamically loaded prompts
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pattern_keys = list(PROMPT_PATTERNS.keys())


print("PATTERN KEYS =>>>>", pattern_keys)
pattern_texts = list(PROMPT_PATTERNS.values())
print("PATTERN TEXTS =>>>>", pattern_keys)
pattern_embeddings = embedding_model.encode(pattern_texts)
