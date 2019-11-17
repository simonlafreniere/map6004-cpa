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

    transposee = mean_matrice.Transposee()
    AAT = transposee * mean_matrice  # l'ordre est inverse car on veut que les colonnes de la matrice soit les categories (agr, min, man..)
    covariances = AAT / (mean_matrice.colonnes - 1)  # S = AAT/n-1
    print("covariances (S):\n" + str(covariances))
    eigenvalues, U = Matrices.Jacobi(covariances)
    print("lambdas (eigenvalues):\n" + str(eigenvalues))
    print("U (eigenvectors):\n" + str(U))


if __name__ == "__main__":
    main()
