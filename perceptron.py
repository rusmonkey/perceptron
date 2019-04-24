import numpy as np


class Perceptron(object):

    def __init__(self, vector_len, threshold=100):
        self.threshold = threshold
        weights = [[8, 1, 4, 7, 2, 6, 9, 5, 3],
                   [7, 2, 5, 8, 1, 4, 7, 2, 6],
                   [1, 0, 3, 7, 2, 5, 4, 3, 5],
                   [9, 2, 3, 4, 3, 5, 1, 0, 3],
                   [1, 7, 2, 5, 1, 8, 1, 4, 7],
                   [6, 9, 2, 4, 3, 5, 7, 2, 5],
                   [5, 1, 0, 3, 7, 2, 5, 4, 3],
                   [4, 1, 8, 1, 4, 7, 2, 6, 9],
                   [4, 3, 5, 1, 0, 3, 7, 2, 5],
                   [9, 2, 3, 5, 8, 1, 4, 7, 2]]
        self.weights = np.array(weights, dtype=np.int32)

    def predict(self, inputs, label):
        summation = np.zeros(10, dtype=np.int32)
        for i in range(0, 10, 1):
            summation[i] = np.sum(np.dot(inputs, self.weights[i]))
        # max_weight = np.amax(summation)
        if np.delete((summation < summation[label]), label).all():
            activation = 1
        else:
            activation = 0
        return activation, np.argwhere(summation > summation[int(label)])

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs, label)
                if prediction[0] == 0:
                    self.weights[int(label)] = np.add(self.weights[int(label)], inputs)
                    for i in prediction[1]:
                        if i != int(label):
                            self.weights[i] = np.subtract(self.weights[i], inputs)
