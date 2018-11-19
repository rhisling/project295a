import json, argparse, time

import tensorflow as tf
from load import load_graph

from flask import Flask, request
from flask_cors import CORS


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/predict',methods=['POST'])
def predict():
    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']


if __name__ == '__main__':
    app.run()
