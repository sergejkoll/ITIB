from neuron import Neuron

if __name__ == '__main__':
    n = Neuron(4.5, 5, 4, 20, 0.1, 10000)
    n.learning()
    n.plot_after_learning()
    n.predict()
    n.plot_predict()
