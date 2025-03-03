import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_similarity(input_text, template_prompts, template_names):
    """
    Calculates similarity scores between the input text and prompt templates.
    Returns a pandas DataFrame with similarity scores.
    """
    # Encode input text and prompt templates
    input_embedding = model.encode([input_text])
    template_embeddings = model.encode(template_prompts)

    # Compute similarity scores
    similarities = cosine_similarity(input_embedding, template_embeddings)[0]

    # Create a DataFrame
    df = pd.DataFrame({
        "Prompt_Keys": template_names,
        "Prompt_Templates": template_prompts,
        "Similarity_Score": np.round(similarities, 4)
    })

    # Sort by highest similarity score
    df = df.sort_values(by="Similarity_Score",
                        ascending=False).reset_index(drop=True)

    return df


def load_prompt_data(file_path):
    """
    Reads the prompt data from a JSON file and extracts template names, prompts, and formats.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        prompt_data = json.load(file)

    # Extract template names, prompts, and formats
    template_names = list(prompt_data.keys())
    template_prompts = [data["prompt_template"]
                        for data in prompt_data.values()]
    prompt_formats = [data["output_format"] for data in prompt_data.values()]

    return template_names, template_prompts, prompt_formats


def extract_best_prompt(input_text, template_prompts, template_names):
    """
    Extracts the best matching prompt template based on similarity calculation.
    Returns the key of the best matching prompt.
    """
    # Calculate similarity scores
    similarities = calculate_similarity(
        input_text, template_prompts, template_names)

    # Find the key of the best matching prompt
    best_match_key = similarities.iloc[0]["Prompt_Keys"]

    return best_match_key


def subset_best_prompt(file_path, best_prompt_key):
    """
    Extracts the best prompt and its format using the identified prompt key.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        prompt_data = json.load(file)

    # Extract the best prompt and format
    best_prompt = prompt_data[best_prompt_key]["prompt_template"]
    best_format = prompt_data[best_prompt_key]["output_format"]

    return best_prompt, best_format


def create_prompt_template_json(best_prompt, best_format, output_file="best_prompt_template.json"):
    """
    Creates a new JSON file containing the best prompt template and output format.
    """
    prompt_template = {
        "prompt_template": best_prompt,
        "output_format": best_format
    }

    # Write to a new JSON file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(prompt_template, file, indent=4)

    print(f"\n\nBest prompt template saved to {output_file}")


# Load prompt data from the JSON file
file_path = "prompt_steering_template.json"
template_names, template_prompts, prompt_formats = load_prompt_data(file_path)

# Example input text
input_text = "suggest gift items for a birthday"

print("\nInput Text: ", input_text, "\n")

# Compute similarity and display the DataFrame
df_similarity = calculate_similarity(
    input_text, template_prompts, template_names)
print("Similarity Results:")
print(df_similarity)

# Compute the best matching prompt template key
best_prompt_key = extract_best_prompt(
    input_text, template_prompts, template_names)

# Extract the best prompt and format
best_prompt, best_format = subset_best_prompt(file_path, best_prompt_key)

# Create a new JSON file with the best prompt template and format
create_prompt_template_json(best_prompt, best_format)

# Display results
print("\nBest Prompt Key:", best_prompt_key)
