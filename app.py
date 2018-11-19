import argparse
import json


import tensorflow as tf
from flask import Flask, request, render_template

from load import load_graph

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

    data = {
        "cpu_frequency": 13.56,
        "cpu_gain": 15.56
    }
    x = []
    for key, value in data:
        x.append(value)

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y1, ({
            x1: [x]
        }))
    json_data = json.dumps()
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)
    x1 = graph.get_tensor_by_name('prefix/Input:0')
    y1 = graph.get_tensor_by_name('prefix/add:0')

    app.run()
