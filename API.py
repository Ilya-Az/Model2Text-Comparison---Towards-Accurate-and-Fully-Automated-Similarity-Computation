import AutoBPMN_AI_Service
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.json
    try:
        results = AutoBPMN_AI_Service.process(body)
        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
