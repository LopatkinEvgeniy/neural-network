from lib import Neuron, Network

network = Network([
    [Neuron(2), Neuron(2), Neuron(2), Neuron(2)],
    [Neuron(4)]
])

training_data = [
    [[0.0, 0.0], [0.0]],
    [[1.0, 0.0], [1.0]],
    [[0.0, 1.0], [1.0]],
    [[1.0, 1.0], [0.0]]
]


def train(learning_rate, delta):
    error = delta * 2
    i = 0
    while error > delta:
        error = 0
        for d in training_data:
            error += abs(network.train_step(d[0], d[1], learning_rate))
        error /= len(training_data)

        if i % 10000 == 0:
            print(i, error)
        i += 1


train(0.1, 0.1)


for d in training_data:
    res = network.process(d[0])
    print(d[1], res)
