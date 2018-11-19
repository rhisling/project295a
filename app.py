import argparse
import json

import tensorflow as tf
from flask import Flask, request

from load import load_graph

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    input: json data of 23 paramteres
    :return: the predicted power
    """
    data = request.data
    dataDict = json.loads(data)
    print(dataDict)
    data = {
        "cpu_frequency": 13.56,
        "no_network_connections": 2,
        "time_spent_user": 15.56,
        "time_spent_system": 23,
        "time_spent_idle": 1,
        "time_spent_io": 13,
        "cpu_percentage": 3,
        "ctx_switches": 3,
        "interrupts": 3,
        "software_interrupts": 23,
        "system_calls": 3,
        "percent_virtual_memory": 2,
        "cached_memory_changed": 2,
        "shared_memory_changed": 3,
        "swap_percentage": 3,
        "swap_in_bytes": 2,
        "swap_out_bytes": 3,
        "bytes_read": 2,
        "bytes_write": 2,
        "bytes_sent": 1,
        "bytes_received": 2,
        "no_processes": 3,
        "rapl_value": 2
    }

    final_res = []
    for i in dataDict:
        x = list()
        x.append(i["cpu_frequency"])
        x.append(i["no_network_connections"])
        x.append(i["time_spent_user"])
        x.append(i["time_spent_system"])
        x.append(i['time_spent_idle'])
        x.append(i["time_spent_io"])
        x.append(i["cpu_percentage"])
        x.append(i["ctx_switches"])
        x.append(i["interrupts"])
        x.append(i["software_interrupts"])
        x.append(i["system_calls"])
        x.append(i["percent_virtual_memory"])
        x.append(i["cached_memory_changed"])
        x.append(i["shared_memory_changed"])
        x.append(i["swap_percentage"])
        x.append(i["swap_in_bytes"])
        x.append(i["swap_out_bytes"])
        x.append(i["bytes_read"])
        x.append(i["bytes_write"])
        x.append(i["bytes_sent"])
        x.append(i["bytes_received"])
        x.append(i["no_processes"])
        x.append(i["rapl_value"])
        final_res.append(x)

    # my_randoms = [random.randrange(1, 101, 1) for _ in range(23)]
    # ##y_out = [[]]
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y1, ({
            x1: final_res
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
