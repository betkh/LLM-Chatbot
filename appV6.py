import json
import requests
import re
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


"""
-----------------------------
------------DONE-------------
-----------------------------

    - proptotying with :
        - various models 
        - LLM interaction approaches (langchain, OLAMA, API requests)
    - prompt engineering with patterns
    - dyanamically selecting the best  prompt handling 
    - accepting user prompt and enriching it
    - context set up for follow up questions ( upto 5 previous chats)
    - tracking chat history 
    - data extraction from response
    - ignored think block







TO DOs:

__________
Front End:
----------

    - ***Top Priority Feature***: 
        - render data viz into front end page (+ some description)
        - generate the embedding vector for each prompt pattern and store it (avoid re-generating)
        - add each prompt pattern to its own separate file

    
    - show user icon = my pic, find better LAMA ICON
    - change title 
    - green circle around lama icon   
    - show progress
    - show loading 
    - drop down menu to select various LLMS such as : deepseek-r1, deepseek-r1:32B, LLAMA2, mistral, 
    - graphic rendering for:
        - MD
        - code 
        - tabular formats 



- how do I write a listener that will render the response from LLM while my back end is processing to render the imediate response in RT
- web-socket protocol 
- constantly poling 
-  API listening 
- updating the DOM


__________
Back End:
----------
    - ***Top Priority Feature***: 

        - implement data visualization function for each of expeted patterns

    
    LOW PRIORITY:
    - enhance prompts further, make it more felxiable and less hardcoded
    - move prompt pattern to a file and read from it
    - modularize the code, avoid "god class/function"
    - dockerize / containerize



_____________
Documentation:
-------------
    - report paper
    - Block Diagrams:
        - high level architecture 
        - prompt engineering & user input text processing using embedding and cosine similarity
        - front end structural architecture
        - front end behavoral architecture (JS)
        - back-end structural oraganization
    - code comments 
    - github documents

____________
Nice To have
------------
 
    - RAG it (Retrival Augmented Gen)
    - DAG it (Automate Data work/flow)
    - AI based prompt engineeing (custom prompts automatically) 
    - host it (blabla.com)


"""


app = Flask(__name__)

# Set up the base URL for the local Ollama API
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_NAME = "llama2"
# MODEL_NAME = "llama3.3"


# Define file paths
CHAT_HISTORY_FILE = "chat_history.json"
EXTRACTED_JSON_FILE = "extracted_emissions_data.json"

# Limit on how many past interactions to keep for context
CONTEXT_LIMIT = 5

# Load sentence transformer for encoding prompts and user queries
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define structured response patterns
PROMPT_PATTERNS = {

    "Transportation_Type_carbon_footprint_analysis":



    """



        You are an AI assistant specialized in carbon footprint analysis of transportation. 
        Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
        (in kg CO2 per passenger km) for different transportation methods. 
        The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows:

        ```json
      [
  {
    "Transportation_Type": "Flight",
    "Description": "Domestic flight (short-haul)",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.25
  },
  {
    "Transportation_Type": "Flight",
    "Description": "International flight (long-haul)",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.15
  },
  {
    "Transportation_Type": "Bus",
    "Description": "Local bus (diesel)",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.1
  },
  {
    "Transportation_Type": "Bus",
    "Description": "Electric bus",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.05
  },
  {
    "Transportation_Type": "Train",
    "Description": "High-speed train (electric)",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.03
  },
  {
    "Transportation_Type": "Driving_Private_Car",
    "Description": "Gasoline car",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.2
  },
  {
    "Transportation_Type": "Driving_Private_Car",
    "Description": "Electric car",
    "Average_Carbon_Footprint_kgCO2_per_passenger_km": 0.06
  }
]```
    """,




    "explanatory_format": """
        You are an AI tutor. Provide a detailed explanation in this format:

        **Concept**: [Define the topic]
        **Explanation**: [Detailed breakdown]
        **Examples**: [Provide real-world examples]
    """,



    "comparative_format": """
        You are an AI assistant specializing in comparisons. Format output as follows:

        - **Aspect 1**: [Describe]
        - **Aspect 2**: [Describe]
        - **Key Differences**: [List]
        - **Final Verdict**: [Summarize]
    """,



    "step_by_step_format": """
        You are an AI guide. Always respond in this step-by-step format:

        1. **Step 1**: [Describe]
        2. **Step 2**: [Describe]
        3. **Step 3**: [Describe]
        4. **Final Step**: [Summarize outcome]
    """,
    "summary_format": """
        You are a structured AI assistant. Format the response as follows:

        - **Summary**: [Brief summary]
        - **Key Points**: [Bullet points]
        - **Conclusion**: [Final thoughts]
    """,


    "car_sizes_Sedan_SUV_Van_Truck":

    """ You are an AI assistant specialized in carbon footprint analysis 
    of various category of cars such as Sedan, SUV, VAN, Truck, etc.
    Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
    (in kg CO2 per mile) for each of these types of vehicles  (Sedan, SUV, VAN, Truck). 

    Quantitative values: 
    values of the key Carbon_Footprint_kgCO2_per_Mile must be numerical and it should be an exact number! 
    avoid giving string values or values with hyphen. Also avoid null values, just provide a number!

    for example: DON'T -> "Carbon_Footprint_kgCO2_per_Mile": "6-8", DO: "Carbon_Footprint_kgCO2_per_Mile": 8

    OUTPUT FORMAT:
    The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows:
    
    ```json
    [
    {"Car_Category": "Sedan", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"},
    {"Car_Category": "SUV", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"},
    {"Car_Category": "VAN", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"},
    {"Car_Category": "Truck", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"}
    ]```

""",

    "car_types_corrosponding_energy_soures_like_Gas_Electric_Hybrid":

    """
    You are an AI assistant specialized in carbon footprint analysis 
    of various types of cars interms of energy source such as Gasoline, Electric, Diesel, Hybrid.
    Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
    (in kg CO2 per mile) for each of these categories of vehicles  (Gasoline, Electric, Diesel, Hybrid). 

    Quantitative values: 
    values of the key Carbon_Footprint_kgCO2_per_Mile must be numerical and it should be an exact number! 
    avoid giving string values or values with hyphen. Also avoid null values, just provide a number!

    for example: DON'T -> "Carbon_Footprint_kgCO2_per_Mile": "6-8", DO: "Carbon_Footprint_kgCO2_per_Mile": 8

    OUTPUT FORMAT:
    The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows:

    ```json 
    [
    {"Car_Type": "Gasoline", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"},
    {"Car_Type": "Electric", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"},
    {"Car_Type": "Diesel", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"},
    {"Car_Type": "Hybrid", "Carbon_Footprint_kgCO2_per_Mile": "[numerical_value]"}
    ]```

""",

    "help_to_choose_best_possible_car_by_size_and_energy_source":

    """
You are an AI assistant specialized in carbon footprint analysis 
of various types of cars interms of a COMBINATIOIN OF 8 CATEGORIES that focus on 
energy source such as Gasoline, Electric, Diesel, Hybrid AS WELL AS type of car such as
(Sedan, SUV, VAN, Truck).

Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
(in kg CO2 per mile) for each of these categories of vehicles  
(Gasoline, Electric, Diesel, Hybrid, Sedan, SUV, VAN, Truck). 

Quantitative values: 
values of the key Carbon_Footprint_kgCO2 must be numerical and it should be an exact number! 
avoid giving string values or values with hyphen. Also avoid null values, just provide a number!

for example: DON'T -> "Carbon_Footprint_kgCO2": "6-8", DO: "Carbon_Footprint_kgCO2_per_Mile": 8

OUTPUT FORMAT:
The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows:

```json
[
    
    {"Car_Type": "Gasoline", "Car_Category": 
    "Sedan", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Gasoline", "Car_Category": 
    "SUV", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Gasoline", "Car_Category":
     "VAN", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Gasoline", "Car_Category": "Truck", "Carbon_Footprint_kgCO2": "[numerical_value]"},

    
    {"Car_Type": "Electric", "Car_Category": 
    "Sedan", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Electric", "Car_Category": 
    "SUV", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Electric", "Car_Category": 
    "VAN", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Electric", "Car_Category": 
    "Truck", "Carbon_Footprint_kgCO2": "[numerical_value]"},

   
    {"Car_Type": "Diesel", "Car_Category": 
    "Sedan", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Diesel", "Car_Category": 
    "SUV", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Diesel", "Car_Category": 
    "VAN", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Diesel", "Car_Category": 
    "Truck", "Carbon_Footprint_kgCO2": "[numerical_value]"},

    
    {"Car_Type": "Hybrid", "Car_Category": 
    "Sedan", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Hybrid", "Car_Category": 
    "SUV", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Hybrid", "Car_Category": 
    "VAN", "Carbon_Footprint_kgCO2": "[numerical_value]"},
    {"Car_Type": "Hybrid", "Car_Category": 
    "Truck", "Carbon_Footprint_kgCO2": "[numerical_value]"}]
    ```

""",

    "co2_emissions_in_various_economic_sectors_in_USA":

    """
    You are an AI assistant specializing in carbon emission data extraction and analysis across various 
    sectors in the United States.

    Your task is to estimate the Total_CO2_Emissions_metric_tons for each year from 2020 to 2024 across 
    six key economic sectors: Energy Sector, Transportation Sector, Industry, Residential Sector,  Commercial Sector, 
    and Agricultural Sector. 

    Quantitative values: 
    values of the key Total_CO2_Emissions_metric_tons must be numerical and it should be an exact number! 
    avoid giving string values or values with hyphen. Also avoid null values, just provide a number!

    for example: DON'T -> "Total_CO2_Emissions_metric_tons": "6-8", DO: "Total_CO2_Emissions_metric_tons": 8

    OUTPUT FORMAT:
    Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
    (in Metric tons) for each of those sectors:
    (Energy, Transportation, Industry, Residential, Commercial, and Agricultural). 

    The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows:


    ```json
    [
    {"Sector": "Energy_Sector", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"Sector": "Transportation_Sector", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"Sector": "Industrial_Sector", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"Sector": "Residential_Sector", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"Sector": "Commercial_Sector", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"Sector": "Agricultural_Sector", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"}
]

""",

    "Co2_emissions_by_US_States":

    """

    You are an AI assistant specializing in carbon emission data extraction and analysis across various 
    States in the United States.

    Your task is to estimate the Total_CO2_Emissions_metric_tons for each year from 2020 to 2024 from the 
    States in the US and list down the top 10 including CA, TX, FL, NY, PA, IL, OH, GA, NC, MI. 

    what states to include :
    Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
    (in Metric tons) for each of those states in the US :
    (CA, TX, FL, NY, PA, IL, OH, GA, NC, MI). Feel free to list other states which are not in the list
    I provided as long as the states are mong the top 10 by co2 emission. 

    Quantitative values: 
    values of the key Total_CO2_Emissions_metric_tons must be numerical and it should be an exact number! 
    avoid giving string values or values with hyphen. Also avoid null values, just provide a number!

    for example: DON'T -> "Total_CO2_Emissions_metric_tons": "6-8", DO: "Total_CO2_Emissions_metric_tons": 8

    Ooutput format:
    The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows:

    ```json
    [
    {"State": "California", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Texas", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Florida", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "New York", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Pennsylvania", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Illinois", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Ohio", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Georgia", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "North Carolina", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"},
    {"State": "Michigan", "Year": "[year value]", "Total_CO2_Emissions_metric_tons": "[numerical_value]"}
]```

""",

    "gift_items_suggestion ":

    """
    You are an AI assistant specializing in carbon emission data extraction and analysis across various 
    gift items which might be given to people for different occasions.

    Your task is to estimate the Total_CO2_Emissions_in_kg for various types of gift items under various categories
    including electronics, fashion, home & living, books, personal care, etc. You must pick the right gift item for 
    the right occasions. Popular occasions are Graduation, Birthdays, Father's Day. Valentines Day, Mother's Day,
    Housewarming, Anniversary, Christmas, ... the list is endless and there is no particular order. 

    What gift items to include :

    virtually anything from smartphones, Handbag, Scarf, Scented Candle Set, Indoor Plant, Bestseller Novel,
    Cookbook, Skincare Set, etc.

    How many suggestions?:
    Suggest 10 gift items at a time. 

    randomness control:
    every time generate a new suggestions because the idea is to suggest as many new ideas as possible to spark 
    interest or help someone buy a gift for any given special occasion.  


    Your ONLY task is to provide a structured JSON response containing estimated CO2 emissions 
    (in kg) for each of those gift items. 

    

    Output format:
    The response MUST be enclosed within triple backticks (```) and structured EXACTLY as follows
    while the keys remain fixed the values can be randomized:
    
    Quantitative values: 
    values of the key Average_CO2_Footprint_kgCO2 must be numerical and it should be an exact number! 
    avoid giving string values or values with hyphen. Also avoid null values, just provide a number!

    for example: DON'T -> "Average_CO2_Footprint_kgCO2": "6-8", DO: "Average_CO2_Footprint_kgCO2": 8
    
    ```json[
    {
        "Gift_Item": "Smartphone",
        "Category": "Electronics",
        "Occasion": "Birthday",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
    },
    {
        "Gift_Item": "Wireless Earbuds",
        "Category": "Electronics",
        "Occasion": "Christmas",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "High-quality wireless earbuds with noise cancellation."
    },

 
    {
        "Gift_Item": "Designer Handbag",
        "Category": "Fashion",
        "Occasion": "Anniversary",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Luxury handbag from a renowned designer."
    },
    {
        "Gift_Item": "Silk Scarf",
        "Category": "Fashion",
        "Occasion": "Mother's Day",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Elegant silk scarf with a unique design."
    },

    
    {
        "Gift_Item": "Scented Candle Set",
        "Category": "Home & Living",
        "Occasion": "Housewarming",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "A set of luxurious scented candles."
    },
    {
        "Gift_Item": "Indoor Plant",
        "Category": "Home & Living",
        "Occasion": "Thank You",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Low-maintenance indoor plant in a decorative pot."
    },


    {
        "Gift_Item": "Bestseller Novel",
        "Category": "Books",
        "Occasion": "Graduation",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Latest bestseller novel by a popular author."
    },
    {
        "Gift_Item": "Cookbook",
        "Category": "Books",
        "Occasion": "Wedding",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Gourmet cookbook with recipes from around the world."
    },

  
    {
        "Gift_Item": "Skincare Set",
        "Category": "Personal Care",
        "Occasion": "Valentine's Day",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Premium skincare set with natural ingredients."
    },
    {
        "Gift_Item": "Perfume",
        "Category": "Personal Care",
        "Occasion": "Father's Day",
        "Average_CO2_Footprint_kgCO2": "[numerical_value]",
        "Description": "Signature fragrance for men."
    }
]```

"""


}

# Precompute embeddings for prompt patterns
pattern_keys = list(PROMPT_PATTERNS.keys())
pattern_texts = list(PROMPT_PATTERNS.values())
pattern_embeddings = embedding_model.encode(pattern_texts)


def save_chat_history(history):
    """ Saves updated chat history to the JSON file. """
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(history, file, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def load_chat_history():
    """ Loads chat history from the JSON file. """
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return empty list if no history exists


def get_chat_context():
    """
    Retrieves the last few chat interactions as context for the bot.
    """
    history = load_chat_history()
    context = history[-CONTEXT_LIMIT:]  # Keep only the last few interactions

    formatted_context = [
        {"role": "system", "content": "You are a helpful AI assistant. Use prior context to answer follow-up questions correctly."}]

    for entry in context:
        formatted_context.append({"role": "user", "content": entry['User']})
        formatted_context.append(
            {"role": "assistant", "content": entry['Bot']})

    return formatted_context


def select_best_prompt(user_query):
    """
    Finds the most relevant prompt pattern for the user's query.
    """
    query_embedding = embedding_model.encode([user_query])
    similarity_scores = cosine_similarity(
        query_embedding, pattern_embeddings)[0]
    best_match_idx = np.argmax(similarity_scores)
    return pattern_texts[best_match_idx]  # Return best-matching prompt


def enrich_prompt(user_input):
    """
    Selects the best formatting instruction and prepends it to the user input.
    """
    best_prompt = select_best_prompt(user_input)
    enriched_prompt = f"{best_prompt}\n\nUser Query: {user_input}"
    return enriched_prompt


def extract_json_from_bot_response(bot_response):
    """
    Extracts JSON data from a bot response.

    Returns:
        list: Extracted structured data as a list of dictionaries, or None if not found.
    """
    match = re.search(r"```json(.*?)```", bot_response, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
        try:
            extracted_json = json.loads(json_text)
            if isinstance(extracted_json, list):
                return extracted_json
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON due to error: {e}")
    return None


def export_data(bot_response):
    """
    Extracts and saves structured data from the bot response into a JSON file.
    """
    extracted_data = extract_json_from_bot_response(bot_response)
    if extracted_data:
        try:
            with open(EXTRACTED_JSON_FILE, "w") as json_file:
                json.dump(extracted_data, json_file, indent=4)
            print(f"Extracted data saved to {EXTRACTED_JSON_FILE}")
        except Exception as e:
            print(f"Error saving extracted data: {e}")
    return extracted_data


def get_chat_response(input_text, model_name, api_url):
    """
    Sends user input to Ollama API, including chat history for context.
    """
    try:
        context = get_chat_context()
        enriched_input = enrich_prompt(input_text)

        # Dynamically modify prompt
        context.append({"role": "user", "content": enriched_input})

        payload = {
            "model": model_name,
            "messages": context
        }

        response = requests.post(api_url, json=payload)

        if response.status_code != 200:
            return Response(json.dumps({"error": "Failed to fetch response from Ollama", "status": response.status_code}),
                            status=response.status_code, content_type='application/json')

        full_response = []
        in_think_block = False

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    token = json_data.get("message", {}).get("content", "")

                    if "<think>" in token:
                        in_think_block = True
                    if "</think>" in token:
                        in_think_block = False
                        token = token.split("</think>", 1)[-1].strip()

                    if not in_think_block and token:
                        full_response.append(token)

                except json.JSONDecodeError:
                    return Response(json.dumps({"error": f"Failed to parse line: {line}"}), 500)

        final_response = "".join(full_response)

        print("\n\nFINAL RESPONSE =>>: \n\n", final_response)

        history = load_chat_history()
        history.append({"User": input_text, "Bot": final_response})
        save_chat_history(history)

        export_data(final_response)

        return Response(final_response, content_type='text/plain')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, content_type='application/json')


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/clear")
def clear_conversation():
    """
    Clears the chat history JSON file.
    """
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump([], file)
        return jsonify({"status": "Conversation cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_chat_history():
    """
    Returns the saved chat history as JSON.
    """
    return jsonify(load_chat_history())


@app.route("/get", methods=["POST"])
def chat():
    """
    Handles user input and returns the chatbot's response.
    """
    input_text = request.form["msg"]
    return get_chat_response(input_text, MODEL_NAME, OLLAMA_API)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
