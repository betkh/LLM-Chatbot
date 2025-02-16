import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(input_text, template_prompts, template_names):
    """
    Calculates similarity scores between the input text and prompt templates.
    Returns a pandas DataFrame with similarity scores.
    """

    # Load sentence-transformers model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = model.encode([input_text])
    template_embeddings = model.encode(template_prompts)
    similarities = cosine_similarity(input_embedding, template_embeddings)[0]

    df = pd.DataFrame({
        "Prompt_Keys": template_names,
        "Prompt_Templates": template_prompts,
        "Similarity_Score": np.round(similarities, 4)
    }).sort_values(by="Similarity_Score", ascending=False).reset_index(drop=True)

    return df


def get_best_prompt(input_text, file_path):
    """
    Finds the best prompt template for the input text and saves it to a JSON file.
    Returns the similarity results, best prompt key, and the contents of best_prompt_template.json.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        prompt_data = json.load(file)

    template_names = list(prompt_data.keys())
    template_prompts = [data["prompt_template"]
                        for data in prompt_data.values()]

    # Calculate similarity and get the best prompt
    df_similarity = calculate_similarity(
        input_text, template_prompts, template_names)
    best_prompt_key = df_similarity.iloc[0]["Prompt_Keys"]
    best_prompt, best_format = prompt_data[best_prompt_key][
        "prompt_template"], prompt_data[best_prompt_key]["output_format"]

    # Save the best prompt to a JSON file
    best_prompt_content = {
        "prompt_template": best_prompt,
        "output_format": best_format
    }
    with open("best_prompt_template.json", "w", encoding="utf-8") as file:
        json.dump(best_prompt_content, file, indent=4)

    print(f"\nBest prompt template saved to best_prompt_template.json")

    # Return similarity results, best prompt key, and the contents of best_prompt_template.json
    return df_similarity, best_prompt_key, best_prompt_content


# Example input text
input_text = "suggest fligts from Chicago to LA"

# Get similarity results and best prompt
df_similarity, best_prompt_key, best_prompt_content = get_best_prompt(
    input_text, "prompt_steering_template.json")

# Display results
print("\nInput Text: ", input_text, "\n")
print("Similarity Results:")
print(df_similarity)
print("\nBest Prompt Key:", best_prompt_key)

print("\nBest Prompt:", best_prompt_content)


input_text_list = []
input_text_list.append(input_text)
input_text_list.append(best_prompt_content)

print("\n\n overall prompt", input_text_list)
