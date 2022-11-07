from flask import Flask, render_template, request
from runner import save_input_to_file, evaluate_input, emotion_indexes_to_lables
import numpy as np
import pickle

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def new():
    return render_template('new.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    input_text = str(request.form['model_input'])
    print(input_text)
    save_input_to_file(input_text)
    prediction = evaluate_input()
    prediction_statement = f"Predicted emotion(s): {', '.join(emotion_indexes_to_lables(prediction[0]))}"

    return render_template('new.html', statement=prediction_statement)


if __name__ == '__main__':
    app.run()
