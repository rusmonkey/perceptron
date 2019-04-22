import numpy as np


class Perceptron(object):

    def __init__(self, vector_len, threshold=100, learning_rate=1):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros((vector_len + 1, 10), dtype=np.int32)

    def predict(self, inputs, label):
        summation = np.zeros(10, dtype=np.int32)
        for i in range(0, 9, 1):
            summation[i] = np.dot(inputs, self.weights[i][1:]) + self.weights[i][0]
        max_weight = np.amax(summation)
        if max_weight > 20 and max_weight == summation[int(label)]:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs, label)
                self.weights[int(label)][1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[int(label)][0] += self.learning_rate * (label - prediction)
