import matrices as Matrices
from excel_file_extration import ExcelExtractor


def main():
    file_name = "data.xlsx"
    extractor = ExcelExtractor(file_name)
    extracted_matrix = extractor.extract_matrix()
    matrice = Matrices.Matrice(len(extracted_matrix), len(extracted_matrix[0]))
    matrice.matrice = extracted_matrix
    print("matrice:\n" + str(matrice))
    mean = Matrices.mean(matrice)
    print("mean:\n" + str(mean))
    mean_matrice = Matrices.substract_mean(matrice)
    print("mean matrice:\n" + str(mean_matrice))
    covariance_matrice = Matrices.covariance_matrix(mean_matrice)
    print("covariance:\n" + str(covariance_matrice))
    t = ""


if __name__ == "__main__":
    main()
