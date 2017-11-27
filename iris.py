import csv
import random
from lib.lib import Neuron, Network

iris_codes = {
    'setosa': [1, 0, 0],
    'versicolor': [0, 1, 0],
    'virginica': [0, 0, 1]
}

iris_dataset = []

for i, row in enumerate(csv.reader(open('data/iris.csv', 'r'))):
    if i == 0:
        continue
    iris_name = row[4]
    iris_data = list(map(lambda n: float(n), row[0:4]))
    iris_dataset.append([iris_data, iris_codes[iris_name]])

random.shuffle(iris_dataset)

validation_data = iris_dataset[0:15]
train_data = iris_dataset[15:]

network = Network([
    [Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4)],
    [Neuron(8), Neuron(8), Neuron(8)]
])

network.train(0.1, 0.1, train_data)

for d in validation_data:
    res = network.process(d[0])
    print(d[1], res)
