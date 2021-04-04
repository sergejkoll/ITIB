from typing import List
import math


class Network:
    def __init__(self, input_vector: List, target: List, N: int,  J: int, M: int, learning_rate: float):
        self.input = input_vector
        self.target = target
        self.N = N
        self.J = J
        self.M = M
        self.learning_rate = learning_rate

        self.hidden_layer_weights = [[0] * (N + 1) for _ in range(J)]
        self.output_layer_weights = [[0] * (J + 1) for _ in range(M)]

    def net_hidden_layer(self, idx) -> float:
        net = self.hidden_layer_weights[idx][0]
        for n in range(self.N):
            net += self.hidden_layer_weights[idx][n + 1] * self.input[n]
        return net

    def net_output_layer(self, idx, input_out_layer: List) -> float:
        net = self.output_layer_weights[idx][0]
        for j in range(self.J):
            net += self.output_layer_weights[idx][j + 1] * input_out_layer[j]
        return net

    @staticmethod
    def f(net) -> float:
        return (1 - math.exp((-1) * net))/(1 + math.exp((-1) * net))

    def derivative(self, net) -> float:
        return 0.5 * (1 - (self.f(net) ** 2))

    def get_epsilon(self, y) -> float:
        epsilon = 0
        for m, target in enumerate(self.target):
            epsilon += (target - y[m]) ** 2
        return math.sqrt(epsilon)

    def get_sum(self, idx, list_delta):
        res = 0
        for m, delta in enumerate(list_delta):
            res += self.output_layer_weights[m][idx] * delta
        return res

    def learning(self):
        epsilon = 1
        K = 0
        while epsilon > 0.0001:
            # first_step
            all_net_in_hidden_layer = []
            all_net_in_out_layer = []
            input_in_out_layer = [0.0] * (self.J + 1)
            out = []
            for j in range(self.J):
                net_in_hidden = self.net_hidden_layer(j)
                all_net_in_hidden_layer.append(net_in_hidden)
                input_in_out_layer[j] = self.f(net_in_hidden)

            for m in range(self.M):
                net_in_out = self.net_output_layer(m, input_in_out_layer)
                all_net_in_out_layer.append(net_in_out)
                out.append(self.f(net_in_out))

            # second_step
            all_out_error = []
            all_hidden_error = []
            for m, net in enumerate(all_net_in_out_layer):
                derivative = self.derivative(net)
                delta = derivative * (self.target[m] - out[m])
                all_out_error.append(delta)

            # Чекнуть get_sum мб j проебываю
            for j, net in enumerate(all_net_in_hidden_layer):
                derivative = self.derivative(net)
                delta = derivative * self.get_sum(j, all_out_error)
                all_hidden_error.append(delta)

            # third_step
            for j in range(self.J):
                self.hidden_layer_weights[j][0] += self.learning_rate * all_hidden_error[j]
                for n in range(self.N):
                    self.hidden_layer_weights[j][n + 1] += self.learning_rate * self.input[n] * all_hidden_error[j]

            for m in range(self.M):
                self.output_layer_weights[m][0] += self.learning_rate * all_out_error[m]
                for j in range(self.J):
                    self.output_layer_weights[m][j + 1] += self.learning_rate * input_in_out_layer[j] * all_out_error[m]

            K += 1
            epsilon = self.get_epsilon(out)
            # print(f"hidden_weights = {self.hidden_layer_weights}")
            # print(f"output_layer = {self.output_layer_weights}")
            result = [round(x, 3) for x in out]
            print(f"y = {result}, E({K}) = {epsilon}")
            if result == self.target:
                break
