import neuralNetwork

if __name__ == '__main__':
    n = neuralNetwork.Network([2], [-0.3], 1, 2, 1, 0.3)
    n.learning()
