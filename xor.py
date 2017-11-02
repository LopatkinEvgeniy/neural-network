def activation(x):
    return 1 if x > 0.5 else 0


hidden_layer = [
    [1, -1],
    [-1, 1]
]

output_layer = [1, 1]


def process(x, y):
    hidden_out_1 = activation(x * hidden_layer[0][0] + y * hidden_layer[0][1])
    hidden_out_2 = activation(x * hidden_layer[1][0] + y * hidden_layer[1][1])
    return activation(hidden_out_1 * output_layer[0] + hidden_out_2 * output_layer[1])


print(process(1, 1))
print(process(0, 1))
print(process(1, 0))
print(process(0, 0))

