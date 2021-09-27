import numpy as np
from flask import Flask, request, jsonify, render_template
import util as util
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


@app.route("/rule", methods=["GET", "POST"])
def pattern_analysis():
    listed = ""
    items_selected = []
    if request.method == "POST":
        item_select1 = request.form.get('item_select1')
        item_select2 = request.form.get('item_select2')
        item_select3 = request.form.get('item_select3')

        if item_select1 != 0:
            items_selected.append(item_select1+ "    ")
            print(items_selected,"aaaaaaaaaaaaaaaaa")
        if item_select2 != 0:
            items_selected.append(item_select2)
        if item_select3 != 0:
            items_selected.append(item_select3)

        item_selected = " , ".join([str(item) for item in items_selected])

        print(item_selected, "bbbbbbbbbbbbbbb")
        listed = util.recommend_product(item_select1, item_select2, item_select3)
        print(listed)

    return render_template('association.html', pattern1=listed,items=item_selected)


if __name__ == '__main__':
    app.run(debug=True)
