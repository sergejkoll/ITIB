from neuron import NeuronWithUnitStepFunction, NeuroneWithSigmoidActivationFunction

if __name__ == '__main__':
    print("Вариант №14")
    print("Задание №1 - Обучение с пороговой функцией активации")
    n = NeuronWithUnitStepFunction([0, 0, 0, 0, 0], 0.3, [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    n.train()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Задание №2 - Обучение с сигмоидальной функцией активации")
    n2 = NeuroneWithSigmoidActivationFunction([0, 0, 0, 0, 0], 0.3, [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    n2.train()
    print("-----------------------------------------------------------------------------------------------------------")
    print("Задание №3 - Обучение с сигмоидальной функцией активации и неполной выборкой")
    n3 = NeuroneWithSigmoidActivationFunction([0, 0, 0, 0, 0], 0.3, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    n3.train_partly()
