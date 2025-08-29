from flask import Flask, request, jsonify
import json
from datetime import datetime
import os

app = Flask(__name__)

# File to save responses
RESPONSES_FILE = "C:/data/nlp/responses.jsonl"

# Ensure file exists
if not os.path.exists(RESPONSES_FILE):
    open(RESPONSES_FILE, 'w').close()

@app.route("/submit_choice", methods=["POST"])
def submit_choice():
    """
    Accept JSON with the following format:
    {
        "prompt": "The text prompt",
        "choice": "response1" | "response2" | "both" | "neither",
        "user_id": "optional_user_identifier"
    }
    """
    print(f"received test")
    try:
        data = request.get_json()
        if not data or "prompt" not in data or "choice" not in data:
            return jsonify({"status": "error", "message": "Invalid payload"}), 400

        # Add timestamp for logging
        data["timestamp"] = datetime.utcnow().isoformat()

        # Append JSON line to file
        with open(RESPONSES_FILE, "a") as f:
            contents    = f.read()
            if not contents:
                contents = []
            else:
                contents    = json.loads(contents)
            
            contents.append(json.dumps(data))
            f.write(contents)

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=2657,debug=True)
