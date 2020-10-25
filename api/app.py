import os
from flask import Flask, request, jsonify

from src.summary_predictor import SummaryPredictor
from src.configs.default_configs import get_cfg_defaults

# Run: cd api | python -m flask run

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do  not use GPU

app = Flask(__name__)


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""

    text = str(load_text())

    cfg = get_cfg_defaults()

    predictor = SummaryPredictor(cfg["model"])
    pred = predictor(text, cfg["model"])

    return jsonify({"pred": str(pred)})


def load_text():
    if request.method == "POST":
        return request.data
    if request.method == "GET":
        print(request.args.get("text"))
        return request.args.get("text")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
