import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron(object):
    def __init__(self, weights):
        self.weights = []
        for i in range(weights):
            self.weights.append(random.uniform(-10, 10))
        self.bias = random.uniform(-10, 10)
        self.last_output = 0

    def process(self, inputs):
        result = 0
        for index, val in enumerate(inputs):
            result += val * self.weights[index]
        result += self.bias
        self.last_output = sigmoid(result)
        return self.last_output


hidden_layer_1 = [Neuron(2), Neuron(2), Neuron(2), Neuron(2)]
out_layer = [Neuron(4)]


def process(x, y):
    hidden_input = [x, y]
    out_input = map(lambda n: n.process(hidden_input), hidden_layer_1)
    return out_layer[0].process(out_input)


training_data = [
    {"x": 0.0, "y": 0.0, "res": 0.0},
    {"x": 1.0, "y": 0.0, "res": 1.0},
    {"x": 0.0, "y": 1.0, "res": 1.0},
    {"x": 1.0, "y": 1.0, "res": 0.0}
]


def train_step(d, learning_rate):
    process(d["x"], d["y"])
    total_error = out_layer[0].last_output - d["res"]

    input_values = map(lambda neuron: neuron.last_output, hidden_layer_1)
    for (i, n) in enumerate(out_layer):
        error = total_error
        weights_delta = error * (n.last_output * (1 - n.last_output))
        for (weight_index, input_val) in enumerate(input_values):
            n.weights[weight_index] -= input_val * weights_delta * learning_rate
        n.bias -= weights_delta * learning_rate

    input_values = [d["x"], d["y"]]
    for (i, n) in enumerate(hidden_layer_1):
        error = 0
        for out_n in out_layer:
            error += out_n.weights[i] * weights_delta

        weights_delta2 = error * (n.last_output * (1 - n.last_output))
        for (weight_index, input_val) in enumerate(input_values):
            n.weights[weight_index] -= input_val * weights_delta2 * learning_rate

        n.bias -= weights_delta2 * learning_rate

    return total_error


def train(learning_rate, delta):
    error = delta * 2
    i = 0
    while error > delta:
        error = 0
        for d in training_data:
            error += abs(train_step(d, learning_rate))
        error /= len(training_data)

        if i % 10000 == 0:
            print(i, error)
        i += 1


train(0.1, 0.1)


for d in training_data:
    res = process(d["x"], d["y"])
    print(d["res"], res)
