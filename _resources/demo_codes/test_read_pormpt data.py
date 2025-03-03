import json


def load_prompt_data(file_path):
    """
    Reads the prompt data from a JSON file and extracts template names and prompts.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        prompt_data = json.load(file)

    # Extract template names and prompt templates
    template_names = list(prompt_data.keys())
    template_prompts = [data["prompt_template"]
                        for data in prompt_data.values()]

    return template_names, template_prompts


# Load prompt data from the JSON file in the current directory
# Ensure the file is in the same directory
file_path = "prompt_steering_template.json"
template_names, template_prompts = load_prompt_data(file_path)

# Print extracted data
# print(template_names)
# print(template_prompts)
