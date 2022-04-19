# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy, copy
from numpy.linalg import matrix_rank


class basis:
    """
    vec1 and vec2 are represented in the standard |0>, |1> basis
    """

    def __init__(self, vecs, vecStrings):
        self.vecs = np.array(vecs)
        self.vecStrings = vecStrings

    @property
    def vec1Str(self):
        return self.vecs[0]

    @property
    def vec2Str(self):
        return self.vecs[1]

    @property
    def vec1(self):
        return self.vecs[0]

    @property
    def vec2(self):
        return self.vecs[1]

    @property
    def vec3(self):
        return self.vecs[2]

    @property
    def vec4(self):
        return self.vecs[3]

    def __str__(self):
        out = "{\n"
        for vec in self.vecs:
            out += arr2SqrtStr(vec)+"\n"
        return out+"}"


_vec1 = 1/np.sqrt(2)*np.array([1, 1], dtype=np.complex128)
_vec2 = 1/np.sqrt(2)*np.array([1, -1], dtype=np.complex128)
_strings = np.array(["+", "-"])
pmBasis = basis(np.array([_vec1, _vec2]), _strings)


_vec1 = np.array([1, 0], dtype=np.complex128)
# print(type(vec1[0]))
_vec2 = np.array([0, 1], dtype=np.complex128)
_strings = np.array(["0", "1"])
oneZeroBasis = basis(np.array([_vec1, _vec2]), _strings)


class psi:
    """
    The Wavefunction
    """

    def __init__(self, vec, basis=oneZeroBasis):
        self.basis = basis

        if isinstance(vec, np.ndarray):
            # This is the current representation
            self.vec = deepcopy(vec).astype(np.complex128)

        # elif isinstance(vec, str):
        #     self.vec = operString2Mat(vec, basis)
        else:
            raise TypeError("oper type not supported")

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.vec = waveChangeBasis(self.vec, self.basis, newBasis)
            self.basis = newBasis

    def __str__(self):
        out = ""
        for i, x in enumerate(self.vec):
            if i == 0:
                sign = ""
            else:
                if x.real < 0:
                    sign = "-"
                else:
                    sign = "  +  "
            out += sign+toSqrtStr(x)+"|"+self.basis.vecStrings[i]+">"
        return out


class psiTensorProd:
    """
    A class for a wavefunction that is a tensor prodcut of other states.
    Using *psi allows for any number of psi values to be passed, for example

    psi1 = lib.psi(np.array([1, 0]))
    psi2 = lib.psi(np.array([0, 1]))
    psi3 = lib.psi( 1/np.sqrt(2)*np.array([1, 1j]))

    psi = lib.psiTensorProd(psi1,psi2,psi3)
    """

    def __init__(self, *psis):

        if len(psis)==1:
            raise ValueError("You should use the regular psi class, not the tensor product version")

        self.psis = np.asarray(psis)

    def preformProd(self, indicies=None):
        """
        Preforms the tensor product on the indicies given

        Parameters
        ------------
        indicies, optional: 2d ndarray, 
            If None, then all elements are taken in the tensor product

            indicies[0] should give the first elements in psis for which the tensor product is taken

            indicies[0] must return consecutive values i.e. 1,2,3, otherwise the tensor product
            gets messed up
        
        Returns
        ---------
        Nothing, this only updates the internal state. If you want to view the state after, just print it
        """
        if isinstance((indicies), type(None)):
            indicies=np.arange(len(self.psis),dtype=int)

        if len(np.shape(indicies))==1: #deal with case when indicies is just passed as an array
            indicies=np.array([indicies])

        if not isConsecutive(indicies):
            raise ValueError("indicies must be consecutive")

        # flatInd = np.ndarray.flatten(indicies)
        firstVals = copy(indicies[:,0])
        psiList = []
        i=0
        while i < len(self.psis):
            if i in firstVals:
                vec = np.array(1, dtype=np.complex128)
                basisList = [] 
                for j in vals:
                    vec = np.kron(vec, self.psis[j].vec)
                    basisList.append(self.psis[j].basis)

                combinedBasis = combineBasis(basisList)
                psiTmp = psi(vec,basis=basis)
                psiList.append(vec)
                i = indicies[i][-1]+1
            else:
                psiList.append(self.psis[i])
                i+=1

    def __str__(self):
        out = ""
        for psi in self.psis:
            out += "( " + psi.__str__() + " )" + u" \u2297  " #  unicode for tensor product
        out = out[:-3]
        return out

def combineBasis(basisList):
    #TODO: impliment this 
    pass

def isConsecutive(indicies):
    """
    Tests if the elements of indicies are arrays with consecutive integers values

    indicies = np.array([np.arange(5,dtype=int),np.arange(3,dtype=int)],dtype=object)
    isConsecutive(indicies)

    indicies = np.array([[1,3,4,5], [1,2,3,4]])
    isConsecutive(indicies)
    """
    for val in indicies:
        difs = val[1:]- val[:-1]
        test = (difs==1).all()
        if not test:
            return False
    return True

class oper:
    """
    Operator Class
    """

    def __init__(self, mat, basis=oneZeroBasis):
        if isinstance(mat, np.ndarray):
            # This is the current representation
            self.mat = deepcopy(mat).astype(np.complex128)

        # elif isinstance(mat, str):
        #     self.mat = operString2Mat(mat, basis)
        else:
            raise TypeError("oper type not supported")

        self.basis = basis  # basis

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.mat = changeBasis(self.mat, self.basis, newBasis)
            self.basis = newBasis

    def __str__(self, whichBasis=False):
        strMat = ""
        for i in range(len(self.mat)):
            strMat += "("
            for j in range(len(self.mat[0])):
                tmp = toSqrtStr(self.mat[i, j])
                if j != len(self.mat[0])-1:
                    tmp += ", "
        # strMat = np.zeros(np.shape(self.mat),dtype=str)

                else:
                    tmp += ")"
                strMat += tmp
            strMat += '\n'
        if whichBasis:
            strMat += "Basis is "
            vecStrings = self.basis.vecStrings
            length = len(vecStrings)
            for i, vecString in enumerate(vecStrings):
                strMat += "|"+vecString+">"
                if i != length-1:
                    strMat += ",  "
        return strMat


def inner(bra, ket):
    """
    inner product of two vectors
    <bra|ket> such that the complex conjugate of a is taken
    """
    return np.dot(np.conjugate(bra), ket)


def changeBasis(mat, basis, newBasis):
    transMat = np.zeros(np.shape(mat), dtype=np.complex128)
    for i in range(len(transMat)):
        for j in range(len(transMat[0])):
            vec1 = basis.vecs[i]
            vec2 = newBasis.vecs[j]
            transMat[i, j] = inner(vec1, vec2)
    conj = np.conjugate(transMat).T
    # print("transMat should be hermetian conjugate, check for zeros")
    # print(conj-transMat)

    tmp = np.dot(conj, mat)
    return np.dot(tmp, transMat)


def waveChangeBasis(vec, basis, newBasis):
    length = len(basis.vec1)
    transMat = np.zeros((length, length), dtype=np.complex128)
    for i in range(len(transMat)):
        for j in range(len(transMat[0])):
            vec1 = basis.vecs[i]
            vec2 = newBasis.vecs[j]
            transMat[i, j] = inner(vec1, vec2)
    conj = np.conjugate(transMat).T
    # print("conj of transMat is \n", transMat)
    # print("transMat should be hermetian conjugate, check for zeros")
    # print(conj-transMat)
    return np.dot(conj, vec)


def arr2SqrtStr(vec):
    """
    Represents some square roots of values as strings that include sqrt(stuff)

    Parameters
    -----------
    vec: ndarray
        The input vector
    Returns
    --------
    out: string
        The representation of that input vector
    """
    rootList = {1/np.sqrt(2): "1/sqrt(2)",
                np.sqrt(3)/2: "sqrt(3)/2"}
    out = "< "
    for val in vec:
        out += toSqrtStr(val)+', '

    return out[:-2]+" >"


def toSqrtStr(val):
    def internal(val2):
        rootList = {1/np.sqrt(2): "1/sqrt(2)",
                    np.sqrt(3)/2: "sqrt(3)/2"}
        for root in rootList.keys():
            if abs(abs(val2)-root) < 10**-5:
                if val2 >= 0:
                    sign = ""
                else:
                    sign = "-"
                return sign+rootList[root]
        return str(np.round(val2, 4))

    if isinstance(val, complex):
        if val.real != 0:
            term1 = internal(val.real)
        else:
            term1 = ""
        if val.imag != 0:
            term2 = internal(val.imag)+"j"
            if term1 != "":
                term2 = " + " + term2
        else:
            term2 = ""
        if term1 == "" and term2 == "":
            term1 = "0"

        return term1+term2

    else:
        return internal(val)
