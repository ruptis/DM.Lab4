import numpy as np


class Perceptron:
    def __init__(self, vector_dimensions, classes_count, learning_rate=1):
        self.vector_dimensions = vector_dimensions
        self.classes_count = classes_count
        self.learning_rate = learning_rate
        self.weights = np.zeros((classes_count, vector_dimensions))
        self.bias = np.zeros(classes_count)
        self.decisive_functions = [DecisiveFunction(self.weights[i], self.bias[i]) for i in range(classes_count)]

    def train(self, data):
        errors = True
        while errors:
            errors = False
            for vector, expected_class in data:
                if self._check_error(vector, expected_class):
                    errors = True
                    self._update_weights(vector, expected_class)
                    self._update_bias(expected_class)
                    self._update_decisive_functions()

    def predict(self, vector):
        return np.argmax([decisive_function(vector) for decisive_function in self.decisive_functions])

    def get_decisive_functions(self):
        return self.decisive_functions

    def _update_weights(self, vector, expected_class):
        for i in range(self.classes_count):
            if i == expected_class:
                self.weights[i] += self.learning_rate * vector
            else:
                self.weights[i] -= self.learning_rate * vector

    def _update_bias(self, expected_class):
        for i in range(self.classes_count):
            if i == expected_class:
                self.bias[i] += self.learning_rate
            else:
                self.bias[i] -= self.learning_rate

    def _update_decisive_functions(self):
        self.decisive_functions = [DecisiveFunction(self.weights[i], self.bias[i]) for i in range(self.classes_count)]

    def _check_error(self, vector, expected_class):
        return self.decisive_functions[expected_class](vector) <= max(
            [self.decisive_functions[i](vector) for i in range(self.classes_count) if i != expected_class])


class DecisiveFunction:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def __call__(self, vector):
        return np.dot(self.weights, vector) + self.bias

    def __repr__(self):
        return (' '.join(
            [f'{"-" if self.weights[i] < 0 else "+"} {abs(self.weights[i])} * x{i}' for i in range(len(self.weights))])
                + f' {"-" if self.bias < 0 else "+"} {abs(self.bias)}')


    def __str__(self):
        return self.__repr__()
