import numpy as np
from flask import Flask, request, jsonify, render_template
import mlflow.sklearn
import json

app = Flask(__name__)

path = "/Users/roland/projects/sandbox/capybara/mlruns/0/bbd6eed0e5e64b17a495c0bcc1bc0e38/artifacts/model/"
model = mlflow.sklearn.load_model(path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = request.json['features']
    print(features)
    features = np.array(features)#.reshape(-1,1)
    print(f"Shape: {features.shape}")

    pred = model.predict(features)
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    #
    # output = round(prediction[0], 2)
    result = dict()
    result['prediction'] = pred.tolist()
    return json.dumps(result)


if __name__ == "__main__":
    app.run(port=6000, debug=True)