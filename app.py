from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/association')
def association_start():
    return render_template('association.html')


@app.route('/classification')
def classification_start():
    return render_template('association.html')


if __name__ == '__main__':
    app.run(debug=True)
