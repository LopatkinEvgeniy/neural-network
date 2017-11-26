import csv
import random

setosa = 'setosa'
setosa_index = 0

versicolor = 'versicolor'
versicolor_index = 1

virginica = 'virginica'
virginica_index = 2

iris_dataset={}
iris_dataset[setosa] = []
iris_dataset[versicolor] = []
iris_dataset[virginica] = []

for i, row in enumerate(csv.reader(open('data/iris.csv', 'r'))):
    if i == 0:
        continue
    iris_dataset[row[4]].append(row[0:4])

random.shuffle(iris_dataset[setosa])
random.shuffle(iris_dataset[versicolor])
random.shuffle(iris_dataset[virginica])

validation_data = []
train_data = []

for i in range(5):
    validation_data.append([iris_dataset[setosa][i], [setosa_index]])
    validation_data.append([iris_dataset[versicolor][i], [versicolor_index]])
    validation_data.append([iris_dataset[virginica][i], [virginica_index]])

for iris_index in iris_dataset:
    for i, data in enumerate(iris_dataset[iris_index]):
        if i < 5:
            continue
        train_data.append([data, [iris_index]])

random.shuffle(validation_data)
random.shuffle(train_data)
