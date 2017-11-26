import csv
import random
from lib.lib import Neuron, Network

setosa = 'setosa'
versicolor = 'versicolor'
virginica = 'virginica'

indexes = {
    setosa: [1, 0, 0],
    versicolor: [0, 1, 0],
    virginica: [0, 0, 1]
}

iris_dataset={}
iris_dataset[setosa] = []
iris_dataset[versicolor] = []
iris_dataset[virginica] = []

for i, row in enumerate(csv.reader(open('data/iris.csv', 'r'))):
    if i == 0:
        continue
    data = list(map(lambda n: float(n), row[0:4]))
    iris_dataset[row[4]].append(data)

random.shuffle(iris_dataset[setosa])
random.shuffle(iris_dataset[versicolor])
random.shuffle(iris_dataset[virginica])

validation_data = []
train_data = []

for i in range(5):
    validation_data.append([iris_dataset[setosa][i], indexes[setosa]])
    validation_data.append([iris_dataset[versicolor][i], indexes[versicolor]])
    validation_data.append([iris_dataset[virginica][i], indexes[virginica]])

for iris_name in iris_dataset:
    for i, data in enumerate(iris_dataset[iris_name]):
        if i < 5:
            continue
        train_data.append([data, indexes[iris_name]])

random.shuffle(validation_data)
random.shuffle(train_data)

network = Network([
    [Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4), Neuron(4)],
    [Neuron(8), Neuron(8), Neuron(8)]
])

network.train(0.1, 0.1, train_data)

for d in validation_data:
    res = network.process(d[0])
    print(d[1], res)
