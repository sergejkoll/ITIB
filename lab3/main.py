from neuron import Neuron

if __name__ == '__main__':
    n = Neuron(4.5, 5, 2, 20, 0.5, 1000)
    n.learning()
    n.predict()
