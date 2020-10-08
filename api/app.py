import os
from flask import Flask, request, jsonify

from ..src.summary_predictor import SummaryPredictor
from ..src.configs.yacs_configs import get_cfg_defaults, add_pretrained


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do  not use GPU

app = Flask(__name__)


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""

    text = request.args.get("text")

    cfg = get_cfg_defaults()
    add_pretrained(cfg)

    predictor = SummaryPredictor(cfg.MODEL)
    pred = predictor(text, cfg)

    return jsonify({'pred': str(pred)})


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
