import pickle
from flask import Flask, request, jsonify, render_template, url_for,app
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    return jsonify({'prediction':output[0]})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    regmodel.predict(final_input)
    output=round(regmodel.predict(final_input)[0],2)
    return render_template('index.html',prediction_text='Predicted Price of the House is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)