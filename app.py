import argparse
import json
import random
import pandas as pd
import tensorflow as tf
from flask import Flask, request

from load import load_graph

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.data.decode("utf-8")
    # if data == "":
    #     params = request.form
    #     x_in = json.loads(params['x'])
    # else:
    #     params = json.loads(data)
    #     x_in = params['x']

    data = {
        "cpu_frequency": 13.56,
        "cpu_gain": 15.56
    }
    x = []
    x.append(data["cpu_frequency"])
    x.append(data["cpu_gain"])
    my_randoms = [random.randrange(1, 101, 1) for _ in range(23)]
    ##y_out = [[]]
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y1, ({
            x1: [my_randoms]
        }))

    json_data = json.dumps({"power": y_out.tolist()})
    return json_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)
    x1 = graph.get_tensor_by_name('prefix/Input:0')
    y1 = graph.get_tensor_by_name('prefix/add:0')

    app.run()
