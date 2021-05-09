from tabulate import tabulate
from typing import List


class TableCreator:
    def __init__(self, headers: List[str], rows: List[List[str]] = None):
        self.rows = [headers]

        if rows is not None:
            for row in rows:
                self.rows.append(row)

        self.column_quantity = len(headers)
        self.headers = headers

    def add_row(self, row: List[str]):
        if len(row) != self.column_quantity:
            raise ValueError

        self.rows.append(row)

    def print_table(self, output, header: str = None, table_fmt: str = None, show_index: bool = None):
        output.write(tabulate(self.rows, headers=header, tablefmt=table_fmt, showindex=show_index))
