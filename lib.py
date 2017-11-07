import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron(object):
    def __init__(self, weights):
        self.output = 0
        self.error = 0
        self.delta = 0
        self.weights = []
        for i in range(weights):
            self.weights.append(random.uniform(-10, 10))
        self.bias = random.uniform(-10, 10)

    def process(self, inputs):
        result = 0
        for i, val in enumerate(inputs):
            result += val * self.weights[i]
        result += self.bias
        self.output = sigmoid(result)
        return self.output


class Network(object):
    def __init__(self, layers):
        self.layers = layers

    def process(self, input_data):
        output_data = []
        for (index, layer) in enumerate(self.layers):
            output_data = list(map(lambda n: n.process(input_data), layer))
            input_data = output_data
        return output_data

    def train_step(self, input_data, expected_result, learning_rate):
        actual_result = self.process(input_data)

        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]

            layer_input = []
            if layer_index == 0:
                layer_input = input_data
            else:
                prev_layer = self.layers[layer_index - 1]
                layer_input = list(map(lambda n: n.output, prev_layer))

            for neuron_index, neuron in enumerate(layer):
                if layer_index == len(self.layers) - 1:
                    neuron.error = actual_result[neuron_index] - expected_result[neuron_index]
                else:
                    neuron.error = 0
                    next_layer = self.layers[layer_index + 1]
                    for i, n in enumerate(next_layer):
                        neuron.error += n.weights[neuron_index] * n.delta

                neuron.delta = neuron.error * neuron.output * (1 - neuron.output)
                neuron.bias -= neuron.delta * learning_rate
                for i, input_val in enumerate(layer_input):
                    neuron.weights[i] -= input_val * neuron.delta * learning_rate

        # FIXME: previous error
        total_error = 0
        for (i, result) in enumerate(actual_result):
            total_error += abs(result - expected_result[i])

        return total_error

    def train(self, learning_rate, delta, training_data):
        error = delta * 2
        i = 0
        while error > delta:
            error = 0
            for d in training_data:
                error += abs(self.train_step(d[0], d[1], learning_rate))
            error /= len(training_data)

            if i % 10000 == 0:
                print(i, error)
            i += 1
