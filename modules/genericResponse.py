from flask import Response, jsonify
import requests
import json


def get_Chat_response_generic(input, modelName, ollama_api):
    try:
        payload = {
            "model": modelName,
            "messages": [{"role": "user", "content": input}]
        }

        response = requests.post(ollama_api, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch response from Ollama", "status": response.status_code}), response.status_code

        def generate():
            for line in response.iter_lines(decode_unicode=True):
                if line:

                    try:
                        json_data = json.loads(line)
                        if "message" in json_data and "content" in json_data["message"]:

                            token = json_data["message"]["content"]
                            print(token)

                            yield token
                    except json.JSONDecodeError:
                        yield f"\nFailed to parse line: {line}"

        return Response(generate(), content_type='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
