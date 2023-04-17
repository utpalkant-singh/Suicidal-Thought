import pickle
import numpy as np
from flask import Flask, request, jsonify

model = pickle.load(open('best_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input as a text string
    text = request.json['text']

    # Use the pre-trained vectorizer to transform the text into features
    text_features = vectorizer.transform([text]).toarray()

    # Use the pre-trained model to make a prediction
    prediction = model.predict(text_features)[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
