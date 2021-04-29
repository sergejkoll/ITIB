import sys

import network as rnn
import symbol

x = symbol.from_string_to_figure([
    '+   +',
    ' + + ',
    '  +  ',
    ' + + ',
    '+   +'], 5, 5
)

y = symbol.from_string_to_figure([
    '+   +',
    '+   +',
    '+++++',
    '    +',
    '+++++'], 5, 5)

z = symbol.from_string_to_figure([
    '+++++',
    '   + ',
    '  +  ',
    ' +   ',
    '+++++'], 5, 5)


def catch_symbol(n: rnn.Network, s: symbol.Figure, is_sync: bool):
    result = n.detect_from_symbol(s, is_sync)
    result.print_figure(sys.stdout)


if __name__ == '__main__':
    net15 = rnn.import_figures([x, y, z], 25, 100)

    catch_symbol(net15, x, True)
    catch_symbol(net15, y, True)
    catch_symbol(net15, z, True)

    w = net15.weights()
    for _, row in enumerate(w):
        matrix_line = '|'
        for i, value in enumerate(row):
            if value >= 0:
                matrix_line += ' '

            matrix_line += str(value)
            if i != len(row):
                matrix_line += ' '
        print(matrix_line + '|')

    print()
    print('FIX')
    print()
    catch_symbol(net15, symbol.from_string_to_figure([
        '+   +',
        ' + + ',
        '     ',
        ' + + ',
        '    +'], 5, 5), True)

    catch_symbol(net15, symbol.from_string_to_figure([
        '+   +',
        '+   +',
        '+  + ',
        '    +',
        '+++++'], 5, 5), True)

    catch_symbol(net15, symbol.from_string_to_figure([
        ' ++++',
        '   + ',
        '  +  ',
        ' +   ',
        '+  ++'], 5, 5), True)
