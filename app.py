import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/association')
def association_start():
    return render_template('association.html')

@app.route('/classification')
def classification_start():
    return render_template('classification.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    return render_template('classification.html', prediction_text='Customer is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
