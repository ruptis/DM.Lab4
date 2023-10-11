from sys import argv

import pandas as pd
from perceptron import Perceptron


def main():
    data = pd.read_csv(argv[1], header=None)
    data = [(tuple(row[:-1]), row[-1]) for row in data.values]
    perceptron = Perceptron(len(data[0][0]), len(set([row[-1] for row in data])))
    perceptron.train(data)
    decisive_functions = perceptron.get_decisive_functions()
    for i in range(len(decisive_functions)):
        print(f'd{i}(x) = {decisive_functions[i]}')


if __name__ == '__main__':
    main()
