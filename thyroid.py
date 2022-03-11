import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    if der == True:  # derivative of the sigmoid
        f = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
    else:  # sigmoid
        f = 1 / (1 + np.exp(-x))

    return f


# Calculate output for one instance
def calculate_output(instance):
    hidden_layer = sigmoid_act(np.dot(w1, instance) + b1)
    output_layer = sigmoid_act(np.dot(w2, hidden_layer) + b2)
    return output_layer


# Reading data from file
with open("ann-train.txt") as f:
    inputs_array = []
    outputs_array = []
    for line in f:
        i = line[:-5]

        if int(line[-4]) == 3:
            outputs_array.append([0, 0, 1])
        elif int(line[-4]) == 2:
            outputs_array.append([0, 1, 0])
        else:
            outputs_array.append([1, 0, 0])

        inputs_array.append([float(x) for x in i.split()])

inputs = np.array(inputs_array)
outputs = np.array(outputs_array)

# Set up the number of perceptron per each layer:
p = 10  # Hidden Layer
q = 3  # Output layer

# Set up the Learning rate
learning_rate = 1 / 5

# 0: Random initialize the relevant data
w1 = 2 * np.random.rand(p, 21) - 1  # Hidden Layer
b1 = 2 * np.random.rand(p) - 1

w2 = 2 * np.random.rand(q, p) - 1  # Output layer
b2 = 2 * np.random.rand(q) - 1

epochs = 150
error = []

# Start the algorithm
for epoch in range(epochs):

    sum_error = 0
    for I in range(3772):

        # 1: input the data
        x = inputs[I]

        # 2: Feed forward
        z1 = sigmoid_act(np.dot(w1, x) + b1)  # Hidden layer
        y = sigmoid_act(np.dot(w2, z1) + b2)  # output layer

        delta_output = (outputs[I] - y) * sigmoid_act(y, der=True)  # output Layer Error
        delta_hidden = np.dot(delta_output, w2) * sigmoid_act(
            z1, der=True
        )  # Hidden Layer Error

        # 3: Updating the weights and bias
        w2 = w2 + np.array([learning_rate * delta_output]).T * z1  # output Layer
        b2 = b2 + learning_rate * delta_output

        w1 = w1 + np.array([learning_rate * delta_hidden]).T * x  # Hidden Layer
        b1 = b1 + learning_rate * delta_hidden

        sum_error += np.mean(abs((outputs[I] - y)))

    average = sum_error / 3772
    if epoch % 10 == 0:
        print("Epoch: " + str(epoch + 1) + " Error: " + str(average))
    error.append(average)


# Test
with open("ann-test.txt") as f:
    inputs_array_test = []
    outputs_array_test = []

    for line in f:
        i = line[:-5]

        if int(line[-4]) == 3:
            outputs_array_test.append([0, 0, 1])
        elif int(line[-4]) == 2:
            outputs_array_test.append([0, 1, 0])
        else:
            outputs_array_test.append([1, 0, 0])

        inputs_array_test.append([float(x) for x in i.split()])


count = 0
for i in range(len(outputs_array_test)):

    o = calculate_output(np.array(inputs_array_test[i]))

    s = [round(o[0]), round(o[1]), round(o[2])]

    if s == outputs_array_test[i]:
        count += 1

print("\nFrom 3428 test subject, the program currectly predicted " + str(count))
print("Accuracy: " + str((count / 3428) * 100) + "\n")


plt.xlabel("Epochs")
plt.ylabel("Error")
plt.plot(error)
plt.show()
