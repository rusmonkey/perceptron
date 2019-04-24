from perceptron import Perceptron
import numpy as np

nill =  [1, 1, 1, 0, 0, 1, 0, 1, 1]
one =   [0, 0, 1, 1, 0, 0, 0, 1, 0]
two =   [0, 1, 1, 0, 0, 0, 1, 0, 1]
three = [0, 1, 0, 1, 1, 0, 1, 0, 0]
four =  [1, 0, 1, 0, 1, 0, 0, 1, 0]
five =  [1, 1, 0, 0, 1, 0, 0, 1, 1]
six =   [0, 0, 0, 1, 1, 1, 0, 1, 1]
seven = [0, 1, 0, 1, 0, 1, 0, 0, 0]
eight = [1, 1, 1, 0, 1, 1, 0, 1, 1]
nine =  [1, 1, 1, 0, 1, 0, 1, 0, 0]

training_input_basic = []
training_input_basic.append(np.array(nill))
training_input_basic.append(np.array(one))
training_input_basic.append(np.array(two))
training_input_basic.append(np.array(three))
training_input_basic.append(np.array(four))
training_input_basic.append(np.array(five))
training_input_basic.append(np.array(six))
training_input_basic.append(np.array(seven))
training_input_basic.append(np.array(eight))
training_input_basic.append(np.array(nine))

training_input = []
labels = np.zeros(0, dtype=np.int32)
for i in range(0, 1000, 1):
    number = np.random.randint(0, 10)
    training_input.append(np.array(training_input_basic[number]))
    labels = np.append(labels, number)
perceptron = Perceptron(9)

perceptron.train(training_input, labels)
print(perceptron.weights)

print("input =", training_input_basic[9])
for i in range(0, 10):
    print(i, perceptron.predict(training_input_basic[i], i)[0], sep=' ')

print(perceptron.predict(training_input_basic[9], 9)[0])
