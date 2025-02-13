from flask import Response, jsonify
import requests
import json


def get_Chat_response_only(input, MODEL_NAME, OLLAMA_API_URL):
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": input}]
        }

        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch response from Ollama", "status": response.status_code}), response.status_code

        def generate():

            new_list = []
            for line in response.iter_lines(decode_unicode=True):
                if line:

                    try:
                        json_data = json.loads(line)
                        token = json_data.get("message", {}).get(
                            "content", "")

                        # isolate the thik block to ignore reasoning essage
                        # Ignore everything inside <think>...</think>
                        if "<think>" in token:
                            in_think_block = True
                        if "</think>" in token:
                            in_think_block = False
                            # Get text after </think>
                            token = token.split("</think>", 1)[-1].strip()

                        if not in_think_block and token:
                            print(token)  # Debugging
                            new_list.append(token.strip())
                            yield token  # Stream only valid output

                        # if "message" in json_data and "content" in json_data["message"]:

                        #     token = json_data["message"]["content"]
                        #     print(token)

                            # yield token
                    except json.JSONDecodeError:
                        yield f"\nFailed to parse line: {line}"
            print(new_list)

        return Response(generate(), content_type='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
