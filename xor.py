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


network.train(0.1, 0.1, training_data)


for d in training_data:
    res = network.process(d[0])
    print(d[1], res)
