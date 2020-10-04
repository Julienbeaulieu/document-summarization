import os
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration


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

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('/home/julien/data-science/nlp-project/weights/model_1_epochs',
                                                       return_dict=True)
    input_ids = tokenizer.encode(text, return_tensors="pt")  # Batch size 1
    outputs = model.generate(input_ids,
                             max_length=150,
                             num_beams=3,
                             repetition_penalty=2.5,
                             length_penalty=1.0,
                             early_stopping=True)

    pred = [tokenizer.decode(g,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for g in outputs][0]

    return jsonify({'pred': str(pred)})


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
