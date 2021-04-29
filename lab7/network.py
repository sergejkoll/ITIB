from typing import List
from symbol import Figure


class Network:
    def __init__(self, x, k: int, quantity: int):
        weights = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    break

                for kk in range(len(x)):
                    weights[i][j] += x[kk][i] * x[kk][j]

                weights[j][i] = weights[i][j]

        self.__weights = weights
        self.__k = k
        self.__max_iterations = quantity

    def weights(self):
        return self.__weights

    def get_out(self, x: List[int], is_sync: bool) -> List[int]:
        out = [0 for _ in range(self.__k)]
        for i, _ in enumerate(x):
            out[i] = x[i]

        prev_out: List[int] = NotImplemented
        check_out: List[int] = NotImplemented

        if is_sync:
            prev_out = [0 for _ in range(self.__k)]
            check_out = prev_out
        else:
            prev_out = out
            check_out = [0 for _ in range(self.__k)]

        for epoch in range(self.__max_iterations):
            if is_sync:
                out, prev_out = prev_out, out
                check_out = prev_out
            else:
                check_out = prev_out.copy()

            for k in range(self.__k):
                net = 0
                for j in range(self.__k):
                    if j == k:
                        continue
                    net += self.__weights[j][k] * prev_out[j]

                if net < 0:
                    out[k] = -1
                elif net > 0:
                    out[k] = 1
                else:
                    out[k] = prev_out[k]

            is_diff = False

            for i in range(self.__k):
                if out[i] != check_out[i]:
                    is_diff = True

            if not is_diff:
                break
        return out

    def detect_from_symbol(self, s: Figure, is_sync: bool) -> Figure:
        return Figure(s.width(), s.height(), self.get_out(s.data(), is_sync))


def import_figures(figures: List[Figure], k: int, max_iterations: int):
    x = [[0] for _ in range(len(figures))]

    for i, figure in enumerate(figures):
        x[i] = figure.data()

    return Network(x, k, max_iterations)



