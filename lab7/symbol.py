from typing import List

Value = '+'
Void = ' '


class Figure:
    def __init__(self, w: int, h: int, vector: List[int]):
        self._width = w
        self._height = h
        self._data = vector

    def height(self) -> int:
        return self._height

    def width(self) -> int:
        return self._width

    def data(self) -> List[int]:
        return self._data

    def to_strings(self) -> list:
        symbol_in_string = [' ' for _ in range(self.height())]
        for j in range(self.height()):
            for i in range(self.width()):
                if self._data[self.height() * i + j] > 0:
                    symbol_in_string[j] += Value
                else:
                    symbol_in_string[j] += Void
        return symbol_in_string

    def print_figure(self, output):
        for j in range(self.height()):
            output.write('\t')
            for i in range(self.width()):
                if self._data[self.height() * i + j] > 0:
                    output.write(Value)
                else:
                    output.write(Void)
            output.write("\n")
        output.write("\n")


def from_string_to_figure(rows: list, w: int, h: int) -> Figure:
    symbol_data = [0 for _ in range(w * h)]

    for i in range(h):
        for j in range(w):
            if rows[i][j] == Value:
                symbol_data[h * j + i] = 1
            else:
                symbol_data[h * j + i] = -1

    return Figure(w, h, symbol_data)
