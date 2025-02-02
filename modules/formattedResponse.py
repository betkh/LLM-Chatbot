from flask import Response, jsonify
import requests
import json


def get_Chat_response_formatted(input, MODEL_NAME, OLLAMA_API_URL):
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": input}]
        }

        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch response from Ollama", "status": response.status_code}), response.status_code

        def generate():
            # Start HTML content
            yield "<html><body>"

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        json_data = json.loads(line)
                        token = json_data.get("message", {}).get("content", "")

                        # Ignore everything inside <think>...</think>
                        if "<think>" in token:
                            in_think_block = True
                        if "</think>" in token:
                            in_think_block = False
                            # Get text after </think>
                            token = token.split("</think>", 1)[-1].strip()

                        if not in_think_block and token:
                            # Wrap the token in HTML tags for rich formatting
                            formatted_token = f"<p>{token}</p>"
                            yield formatted_token

                    except json.JSONDecodeError:
                        yield f"<p>Failed to parse line: {line}</p>"

            # End HTML content
            yield "</body></html>"

        # Return the response with HTML content type
        return Response(generate(), content_type='text/html')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
