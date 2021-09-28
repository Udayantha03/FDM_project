import numpy as np
from flask import Flask, request, jsonify, render_template
import util as util
import pickle
import model as md

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


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    fields = []
    if request.method == "POST":
        fields_1 = request.form.get('gender')
        fields_2 = request.form.get('car')
        fields_3 = request.form.get('realty')
        fields_4 = request.form.get('child')
        fields_5 = request.form.get('income')
        fields_6 = request.form.get('income_type')
        fields_7 = request.form.get('educate')
        fields_8 = request.form.get('fam_stat')
        fields_9 = request.form.get('hou_type')
        fields_10 = request.form.get('age')
        fields_11 = request.form.get('experience')
        fields_12 = request.form.get('occ')

        fields_13 = request.form.get('paid_off')
        fields_14 = request.form.get('past_dues')
        fields_15 = request.form.get('no_loan')

        prediction = md.predict_classification(fields_1, fields_2, fields_3, fields_4, fields_5, fields_6, fields_7, fields_8,
                                               fields_9, fields_10, fields_11, fields_12, fields_13, fields_14, fields_15,
                                               )

        if prediction == 1:
            output = 'GOOD'
        else:
            output = 'BAD'

    return render_template('classification.html', prediction_text='Customer is {}'.format(output))


@app.route("/rule", methods=["GET", "POST"])
def pattern_analysis():
    listed = ""
    items_selected = []
    if request.method == "POST":
        item_select1 = request.form.get('item_select1')
        item_select2 = request.form.get('item_select2')
        item_select3 = request.form.get('item_select3')

        if item_select1 != "0":
            items_selected.append(item_select1+ "    ")
        if item_select2 != "0":
            items_selected.append(item_select2)
        if item_select3 != "0":
            items_selected.append(item_select3)

        item_selected = " , ".join([str(item) for item in items_selected])
        listed = util.recommend_product(item_select1, item_select2, item_select3)

    return render_template('association.html', pattern1=listed,items=item_selected)


if __name__ == '__main__':
    app.run(debug=True)
