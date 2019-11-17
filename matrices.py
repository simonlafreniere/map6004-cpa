from numbers import Number
import random as rd
import time
import numpy as np


class Matrice:

    # constructeur
    def __init__(self, l, c=None, fill=0.0):
        self.lignes = l

        # matrice carree ?
        if c is None:
            self.colonnes = l
        else:
            self.colonnes = c

        # cree une matrice lxc avec des fill
        self.matrice = [[fill] * self.colonnes for i in range(self.lignes)]

    def __str__(self):
        result = ""
        for line in self.matrice:
            result += str(line) + "\n"
        return result

    '''
        operateurs arithmetiques (+,-,*,/)
    '''

    # +
    def __add__(self, other):
        try:
            if not isinstance(other, Matrice):
                raise TypeError('Type Matrice requis')

            # meme dimensions
            if self.colonnes != other.colonnes or self.lignes != other.lignes:
                raise ValueError('Les matrices doivent etre de meme format..')

            reponse = Matrice(self.lignes, self.colonnes)
            for i in range(self.lignes):
                for j in range(self.colonnes):
                    reponse.matrice[i][j] = self.matrice[i][j] + other.matrice[i][j]
            return reponse

        except (TypeError, ValueError) as err:
            print(err.message)
            return None

    # -
    def __sub__(self, other):
        try:
            if not isinstance(other, Matrice):
                raise TypeError('Type Matrice requis')

            if self.colonnes != other.colonnes or self.lignes != other.lignes:
                raise ValueError('Les matrices doivent etre de meme format..')

            reponse = Matrice(self.lignes, self.colonnes)
            for i in range(self.lignes):
                for j in range(self.colonnes):
                    reponse.matrice[i][j] = self.matrice[i][j] - other.matrice[i][j]
            return reponse

        except (TypeError, ValueError) as err:
            print(err.message)
            return None

    # *
    def __mul__(self, other):
        try:
            # multiplication scalaire
            if isinstance(other, Number):
                reponse = Matrice(self.lignes, self.colonnes)
                for i in range(self.lignes):
                    for j in range(self.colonnes):
                        reponse.matrice[i][j] = self.matrice[i][j] * other
                return reponse

            # les 2 sont des matrices?
            if not isinstance(other, Matrice):
                raise TypeError('Type Matrice ou Number requis')

            # Alxp * Bpxc
            if self.colonnes != other.lignes:
                raise ValueError('Toutes les multiplications doivent respecter la contrainte: Alxp * Bpxc..')

            reponse = Matrice(self.lignes, other.colonnes)
            # parcours de la matrice reponseante
            for i in range(self.lignes):
                time.sleep(0.1)  # slow it down sinon c'est trop vite et la diff est en milisecondes..
                for j in range(other.colonnes):
                    # ligne de m multiplie colonne de n
                    for k in range(self.colonnes):
                        reponse.matrice[i][j] += self.matrice[i][k] * other.matrice[k][j]
            return reponse

        except (ValueError, TypeError) as err:
            print(err.message)
            return None

    '''
    2*A == A*2
    faux pour matrices A*B != B*A
    on assume que __mul__ sera appelee..
    '''
    __rmul__ = __mul__

    # /
    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        try:
            self.makeFloat()

            if isinstance(other, Number):
                if other == 0.0:
                    raise ZeroDivisionError('Division par zero..')

                other = float(other)
                reponse = Matrice(self.lignes, self.colonnes)
                for i in range(self.lignes):
                    for j in range(self.colonnes):
                        reponse.matrice[i][j] = self.matrice[i][j] / float(other)
                return reponse

            if not isinstance(other, Matrice):
                raise TypeError('Type Matrice ou Number requis')

            if not other.estCarree():
                raise ValueError('Matrices carrees seulement')

            other.makeFloat()
            # A/B ==> A*BInverse  // c'est ce qui se raproche le plus d'une division matricielle.. qu'on m'a dit..
            if other.estInversible():
                return self * other.Inverse()
            else:
                raise ValueError('La matrice B \'(A/B)\' n\'est pas inversible.. ')

        except (ZeroDivisionError, TypeError, ValueError) as err:
            print(err.message)
            return None

    def randomFilling(self, start=0, end=25):
        for i in range(self.lignes):
            for j in range(self.colonnes):
                self.matrice[i][j] = rd.randint(start, end)

    '''
    operations matricielles
    '''

    # valeurabsolue pour eviter les fausses diagonales vides..
    def Trace(self, valeurabsolue=False):
        try:
            if not self.estCarree():
                raise ValueError('Matrices carrees seulement')

            reponse = 0
            for i in range(self.lignes):
                if valeurabsolue:
                    reponse += abs(self.matrice[i][i])
                else:
                    reponse += self.matrice[i][i]
            return reponse

        except ValueError as err:
            print(err.message)
            return None

    def estCarree(self):
        if self.colonnes == self.lignes:
            return True
        return False

    def estReguliere(self):
        if self.Determinant() != 0:
            return True
        return False

    def Determinant(self):
        try:
            self.makeFloat()

            if not self.estCarree():
                raise ValueError('Matrices carrees seulement')

            # cas de base
            if self.lignes == 1:
                return self.matrice[0][0]
            if self.lignes == 2:
                return self.matrice[0][0] * self.matrice[1][1] - self.matrice[0][1] * self.matrice[1][0]

            # else if diagonale ou triangulaire
            if self.estTriangulaire():
                reponse = self.matrice[0][0]
                for i in range(1, self.lignes):
                    for j in range(1, self.colonnes):
                        if i == j:
                            reponse *= self.matrice[i][j]
                return reponse

            # else
            i = 0  # on choisit la premiere ligne..
            reponse = 0
            for j in range(self.colonnes):
                m = Matrice(self.colonnes - 1)
                # nouvelle matrice self moins ligne i, colonne j
                ligne = 1
                for k in range(m.lignes):
                    colonne = 0
                    for l in range(m.colonnes):
                        if j == l:
                            colonne += 1
                        m.matrice[k][l] = self.matrice[ligne][colonne]
                        colonne += 1
                    ligne += 1
                # determiner le signe
                signe = self.matrice[i][j]
                if (i + j) % 2 != 0:
                    signe *= -1
                reponse += signe * m.Determinant()
            return reponse

        except ValueError as err:
            print(err.message)
            return None

    def Inverse(self):
        try:
            self.makeFloat()

            if not self.estCarree():
                raise ValueError('Matrices carrees seulement')

            if not self.estReguliere():
                raise ZeroDivisionError('Le determinant ne peut etre zero pour la division..')

            if self.estDiagonale():
                reponse = Matrice(self.lignes)
                for i in range(self.lignes):
                    if self.matrice[i][i] != 0:
                        reponse.matrice[i][i] = 1 / self.matrice[i][i]
                    else:
                        reponse.matrice[i][i] = 0
                return reponse

            reciprocal = self.Determinant()
            if self.lignes == 2:
                m = Matrice(2)
                m.matrice[0][0] = self.matrice[1][1]
                m.matrice[1][1] = self.matrice[0][0]
                m.matrice[0][1] = self.matrice[0][1] * -1
                m.matrice[1][0] = self.matrice[1][0] * -1
                return m / reciprocal

            comatriceT = self.CoMatrice().Transposee()
            return comatriceT / reciprocal

        except (ValueError, ZeroDivisionError) as err:
            print(err.message)
            return None

    def CoMatrice(self):
        try:
            if not self.estCarree():
                raise ValueError('Matrices carrees seulement')

            if self.lignes == 1:
                return self

            reponse = Matrice(self.lignes)
            for i in range(self.lignes):
                for j in range(self.colonnes):
                    m = Matrice(self.lignes - 1)
                    ligne = 0
                    for k in range(m.lignes):
                        colonne = 0
                        if ligne == i:
                            ligne += 1
                        for l in range(m.colonnes):
                            if colonne == j:
                                colonne += 1
                            m.matrice[k][l] = self.matrice[ligne][colonne]
                            colonne += 1
                        ligne += 1

                    reponse.matrice[i][j] = m.Determinant()
                    # determiner le signe
                    if (i + j) % 2 != 0:  # and reponse.matrice[i][j]!=0:
                        reponse.matrice[i][j] *= -1

            return reponse

        except ValueError as err:
            print(err.message)
            return None

    def estDiagonale(self):
        try:
            if not self.estCarree():
                raise ValueError('Matrices carrees seulement')

            for i in range(self.lignes):
                for j in range(self.colonnes):
                    if i != j and self.matrice[i][j] != 0:
                        return False

            # diagonale vide 
            if self.Trace(True) == 0:
                raise ValueError('Matrice vide..')

            return True

        except ValueError as err:
            print(err.message)
            return None

    def estTriangulaire(self, sens=None, stricte=False):
        try:
            if not self.estCarree():
                raise ValueError('Matrices carrees seulement')

            inferieure = False
            superieure = False
            for i in range(self.lignes):
                for j in range(self.colonnes):
                    if i < j and self.matrice[i][j] != 0:
                        inferieure = True
                    elif i > j and self.matrice[i][j] != 0:
                        superieure = True

            diagonaleVide = self.Trace(True) == 0
            if not inferieure and not superieure and diagonaleVide:
                raise ValueError('Matrice vide..')

            if stricte and not diagonaleVide:
                return False

            # xor
            if superieure != inferieure:
                if sens != None:
                    if (sens == "inferieure" or sens == "i") and not inferieure:
                        return False
                    elif (sens == "superieure" or sens == "s") and not superieure:
                        return False
                    else:
                        raise ValueError(
                            "Seules les entrees suivantes sont acceptees:\n 'inferieure', 'i', 'superieure', 's'")
                return True
            return False

        except ValueError as err:
            print(err.message)
            return None

    def estInversible(self):
        if not self.estCarree():
            return False
        if self.Determinant() == 0:
            return False
        return True

    def Transposee(self):
        if self.lignes == 1:
            return self

        reponse = Matrice(self.colonnes, self.lignes)
        for i in range(self.lignes):
            for j in range(self.colonnes):
                reponse.matrice[j][i] = self.matrice[i][j]
        return reponse

    # pour jacobi
    def delta(self, other):
        try:
            if not isinstance(other, Matrice):
                raise ValueError('Matrice attendue')
            if self.lignes != other.lignes or self.colonnes != other.colonnes:
                raise ValueError('Matrices de meme dimensions requises')
            if self.colonnes != 1 or other.colonnes != 1:
                raise ValueError('Vecteur requis (\'Matrice colonne, xLignes 0Colonne\')')
            diff = 0
            for i in range(self.lignes):
                diff += abs(self.matrice[i][0] - other.matrice[i][0])

            return diff / self.lignes

        except ValueError as err:
            print(err.message)
            return None

    # strictement dominante diagonalement 
    def estSDD(self):
        for i in range(self.lignes):
            a = 0
            x = 0
            for j in range(self.colonnes):
                if i == j:
                    a = self.matrice[i][j]
                else:
                    x += abs(self.matrice[i][j])
            if not a > x:
                return False
        return True

    def makeFloat(self):
        for i in range(self.lignes):
            for j in range(self.colonnes):
                self.matrice[i][j] = float(self.matrice[i][j])


def mean(matrice):
    mean_vector = []
    for col in range(matrice.colonnes):
        total = 0
        for ligne in range(matrice.lignes):
            total += matrice.matrice[ligne][col]
        mean_vector.append(total / matrice.lignes)
    return mean_vector


def mean_matrice(matrice):
    result = Matrice(matrice.lignes, matrice.colonnes)
    mean_vector = mean(matrice)
    for ligne in range(len(matrice.matrice)):
        result.matrice[ligne] = mean_vector
    return result


def substract_mean(matrice):
    return matrice - mean_matrice(matrice)


def Identite(dimension):
    reponse = Matrice(dimension)
    for i in range(dimension):
        reponse.matrice[i][i] = 1
    return reponse


def MultiplieXMatrices(matrices):
    try:
        # matrices doit etre un dictionnaire..
        if not isinstance(matrices, dict):
            raise TypeError('Les matrices doivent etre stockees dans un dictionnaire..')

        # au moins 2 matrices
        if len(matrices) < 2:
            raise ValueError('Il faut au moins 2 matrices..')

        # toutes sont Alxp * Bpxc
        for i in range(len(matrices) - 1):
            if not isinstance(matrices[i], Matrice) or not isinstance(matrices[i + 1], Matrice):
                raise TypeError('Tous les elements doivent etre de type Matrice..')
            if matrices[i].colonnes != matrices[i + 1].lignes:
                raise ValueError('Toutes les multiplications doivent respecter la contrainte: Alxp * Bpxc..')

        # peupler d
        d = [0] * (len(matrices) + 1)
        d[0] = matrices.get(0).lignes
        d[1] = matrices.get(0).colonnes
        for i in range(1, len(matrices)):
            d[i + 1] = matrices.get(i).colonnes

        # avant parantheses
        csp = 0
        for i in range(len(d) - 2):
            csp += d[i] * d[i + 1] * d[i + 2]
        print('Cout sans parantheses: ' + str(csp))

        # recoit et evalue une string de format ((A*B)*(C*D))*(E*F)...
        return eval(CalculeMeilleurOrdreParantheses(d))

    except (ValueError, TypeError) as err:
        print(err.message)
        return None


def CalculeMeilleurOrdreParantheses(d):
    size = len(d) - 1
    # tableau des couts
    couts = [[None] * size for i in range(size)]
    separation = [[None] * size for i in range(size)]

    # etapes
    for etape in range(size):
        for i in range(size - etape):
            if etape == 0:
                couts[i][i] = 0
            elif etape == 1:
                couts[i][i + 1] = d[i] * d[i + 1] * d[i + 2]
                separation[i][i + 1] = i + 1
            else:
                minimum = -1
                # les cas possibles: (M11+M24+d0d1d4 / M12+M34+d0d2d4 / ...
                for k in range(i, i + etape):
                    least = couts[i][k] + couts[k + 1][i + etape] + d[i] * d[k + 1] * d[i + etape + 1]
                    if minimum == -1:
                        minimum = least
                        separation[i][i + etape] = k + 1
                    if least < minimum:
                        minimum = least
                        separation[i][i + etape] = k + 1
                couts[i][i + etape] = minimum

    # on formatte la string pour permettre l'evaluation..
    parenthesis_order = StringFormatParenthesageMinimal(separation, 0, size - 1)
    parenthesis_order = parenthesis_order.replace(' m', '*m')
    parenthesis_order = parenthesis_order.replace(' ', '')
    parenthesis_order = parenthesis_order.replace(')m', ')*m')
    parenthesis_order = parenthesis_order.replace(')(', ')*(')
    print('Meilleur ordre:')
    print(parenthesis_order)
    print('Cout: ' + str(couts[0][size - 1]))
    return parenthesis_order


def StringFormatParenthesageMinimal(l, i, j):
    if i == j:
        return "matrices.get(" + str(i) + ") "
    else:
        reponse = "("
        reponse += StringFormatParenthesageMinimal(l, i, l[i][j] - 1)
        reponse += StringFormatParenthesageMinimal(l, l[i][j], j)
        reponse += ")"
        return reponse


def Jacobi(A):
    if not A.estCarree():
        raise ValueError('La matrice doit etre carree..')
    n = A.colonnes  # matrice carree
    maxit = 100  # nombre d'iterations maximal
    eps = 1.0e-15  # niveau d'acuitee
    pi = np.pi
    ev = Matrice(1, n)  # initialisation des eigenvalues
    U = Matrice(n)  # initialisation des eigenvector
    for i in range(0, n):
        U.matrice[i][i] = 1.0

    for t in range(0, maxit):
        s = 0  # compute sum of off-diagonal elements in A(i,j)
        for i in range(0, n):
            s = s + np.sum(np.abs(A.matrice[i][(i + 1):n]))
        if s < eps:  # diagonal form reached
            for i in range(0, n):
                ev.matrice[0][i] = A.matrice[i][i]
            break
        else:
            limit = s / (n * (n - 1) / 2.0)  # average value of off-diagonal elements
            for i in range(0, n - 1):  # loop over lines of matrix
                for j in range(i + 1, n):  # loop over columns of matrix
                    if np.abs(A.matrice[i][j]) > limit:  # determine (ij) such that |A(i,j)| larger than average
                        # value of off-diagonal elements
                        denom = A.matrice[i][i] - A.matrice[j][j]  # denominator of Eq. (3.61)
                        if np.abs(denom) < eps:
                            phi = pi / 4  # Eq. (3.62)
                        else:
                            phi = 0.5 * np.arctan(2.0 * A.matrice[i][j] / denom)  # Eq. (3.61)
                        si = np.sin(phi)
                        co = np.cos(phi)
                        for k in range(i + 1, j):
                            store = A.matrice[i][k]
                            A.matrice[i][k] = A.matrice[i][k] * co + A.matrice[k][j] * si  # Eq. (3.56)
                            A.matrice[k][j] = A.matrice[k][j] * co - store * si  # Eq. (3.57)
                        for k in range(j + 1, n):
                            store = A.matrice[i][k]
                            A.matrice[i][k] = A.matrice[i][k] * co + A.matrice[j][k] * si  # Eq. (3.56)
                            A.matrice[j][k] = A.matrice[j][k] * co - store * si  # Eq. (3.57)
                        for k in range(0, i):
                            store = A.matrice[k][i]
                            A.matrice[k][i] = A.matrice[k][i] * co + A.matrice[k][j] * si
                            A.matrice[k][j] = A.matrice[k][j] * co - store * si
                        store = A.matrice[i][i]
                        A.matrice[i][i] = A.matrice[i][i] * co * co + 2.0 * A.matrice[i][j] * co * si + A.matrice[j][j] * si * si  # Eq. (3.58)
                        A.matrice[j][j] = A.matrice[j][j] * co * co - 2.0 * A.matrice[i][j] * co * si + store * si * si  # Eq. (3.59)
                        A.matrice[i][j] = 0.0  # Eq. (3.60)
                        for k in range(0, n):
                            store = U.matrice[k][j]
                            U.matrice[k][j] = U.matrice[k][j] * co - U.matrice[k][i] * si  # Eq. (3.66)
                            U.matrice[k][i] = U.matrice[k][i] * co + store * si  # Eq. (3.67)
    return ev, U
