import xlrd


# basic lecture excel file
class ExcelExtractor:
    def __init__(self, file):
        wb = xlrd.open_workbook(file)
        self.sheet = wb.sheet_by_index(0)
        self.matrice = []

    # exclude ligne 1 et col 1 (noms)
    # ajoute totals
    def extract_matrix(self):
        for i in range(1, self.sheet.nrows):
            row = []
            for j in range(1, self.sheet.ncols):
                row.append(self.sheet.cell_value(i, j))
            self.matrice.append(row)
        return self.matrice
