import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import mlflow.sklearn

if __name__ == "__main__":
    path = "/Users/roland/projects/sandbox/capybara/mlruns/0/bbd6eed0e5e64b17a495c0bcc1bc0e38/artifacts/model/"
    model = mlflow.sklearn.load_model(path)

    input = [[7.4,	0.7,	0,	1.9,	0.076,	11,	34,	0.9978,	3.51,	0.56,	9.4]]
    pred = model.predict(input)
    print(pred)