# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy, copy
from numpy.linalg import matrix_rank

def dentangle(psi):
    """
    If psi is an element of a tensor product of two hilbert spaces, H1 and H2,
    then if possible, this determines the vectors H1 and H2 independently which when 
    tensored together make psi. Based off this post:

    https://math.stackexchange.com/questions/2226935/what-would-be-a-criteria-to-discover-if-a-state-is-a-tensor-product-or-not
    """
    vec = psi.vec
    l = int(np.sqrt(len(vec)))
    mat = np.zeros((l,l), dtype=np.complex128)
    for i in range(l):
        for j in range(l):
            mat[i,j]=vec[i*l+j]  # TODO: check this, it could be wrong

    R = matrix_rank(mat)
    if R!=1:
        raise ValueError("Cannot disentangle")
    #TODO finish this function

class basis:
    """
    vec1 and vec2 are represented in the standard |0>, |1> basis
    """

    def __init__(self, vecs, vecStrings):
        self.vecs=np.array(vecs)
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
        out="{\n"
        for vec in self.vecs:
            out+=arr2SqrtStr(vec)+"\n"
        return out+"}"


_vec1=1/np.sqrt(2)*np.array([1, 1],dtype=np.complex128)
_vec2=1/np.sqrt(2)*np.array([1, -1],dtype=np.complex128)
_strings = np.array(["+","-"])
pmBasis = basis(np.array([_vec1,_vec2]),_strings)


_vec1=np.array([1, 0],dtype=np.complex128)
# print(type(vec1[0]))
_vec2=np.array([0, 1],dtype=np.complex128)
_strings = np.array(["0","1"])
oneZeroBasis = basis(np.array([_vec1,_vec2]),_strings)

class psi:
    """
    The Wavefunction
    """
    def __init__(self, vec, basis=oneZeroBasis):
        self.basis = basis

        if isinstance(vec, np.ndarray):
            self.vec = deepcopy(vec).astype(np.complex128)  # This is the current representation

        elif isinstance(vec, str):
            self.vec = operString2Mat(vec, basis)
        else:
            raise TypeError("oper type not supported")

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.vec = waveChangeBasis(self.vec, self.basis, newBasis)
            self.basis = newBasis

    def __str__(self):
        out=""
        for i, x in enumerate(self.vec):
            if i==0:
                sign=""
            else:
                if x.real<0:
                    sign="-"
                else:
                    sign="  +  "
            out += sign+toSqrtStr(x)+"|"+self.basis.vecStrings[i]+">"
        return out

class oper:
    """
    Operator Class
    """
    def __init__(self, mat, basis=oneZeroBasis):
        if isinstance(mat, np.ndarray):
            self.mat = deepcopy(mat).astype(np.complex128)  # This is the current representation

        elif isinstance(mat, str):
            self.mat = operString2Mat(mat, basis)
        else:
            raise TypeError("oper type not supported")

        self.basis = basis  # basis

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.mat = changeBasis(self.mat, self.basis, newBasis)
            self.basis = newBasis

    def __str__(self, whichBasis=False):
        #TODO: impliment whichBasis statement
        # strMat = np.zeros(np.shape(self.mat),dtype=str)
        strMat = ""
        for i in range(len(self.mat)):
            strMat+="("
            for j in range(len(self.mat[0])):
                tmp = toSqrtStr(self.mat[i,j])
                if j!=len(self.mat[0])-1:
                    tmp+=", "
                else:
                    tmp+=")"
                strMat+=tmp
            strMat+='\n'
        if whichBasis:
            strMat+="Basis is "
            vecStrings = self.basis.vecStrings
            l = len(vecStrings)
            for i, vecString in enumerate(vecStrings):
                strMat+="|"+vecString+">"
                if i!=l-1:
                    strMat+=",  "
        return strMat

def inner(bra, ket):
    """
    inner product of two vectors
    <bra|ket> such that the complex conjugate of a is taken
    """
    return np.dot(np.conjugate(bra),ket)

def operString2Mat(stringOper, basis):
    pass


def changeBasis(mat, basis, newBasis):
    transMat = np.zeros(np.shape(mat),dtype=np.complex128)
    for i in range(len(transMat)):
        for j in range(len(transMat[0])):
            vec1 = basis.vecs[i]
            vec2 = newBasis.vecs[j]
            transMat[i,j] = inner(vec1,vec2)
    conj = np.conjugate(transMat).T
    # print("transMat should be hermetian conjugate, check for zeros")
    # print(conj-transMat)

    tmp = np.dot(conj,mat) 
    return np.dot(tmp,transMat)

def waveChangeBasis(vec, basis, newBasis):
    l = len(basis.vec1)
    transMat = np.zeros((l,l),dtype=np.complex128)
    for i in range(len(transMat)):
        for j in range(len(transMat[0])):
            vec1 = basis.vecs[i]
            vec2 = newBasis.vecs[j]
            transMat[i,j] = inner(vec1,vec2)
    conj = np.conjugate(transMat).T
    # print("conj of transMat is \n", transMat)
    # print("transMat should be hermetian conjugate, check for zeros")
    # print(conj-transMat)
    return np.dot(conj,vec)

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
    rootList = {1/np.sqrt(2):"1/sqrt(2)",
        np.sqrt(3)/2:"sqrt(3)/2"}
    out = "< "
    for val in vec:
        out+=toSqrtStr(val)+', '

    return out[:-2]+" >"

def toSqrtStr(val):
    def internal(val2):
        rootList = {1/np.sqrt(2):"1/sqrt(2)",
                    np.sqrt(3)/2:"sqrt(3)/2"}
        for root in rootList.keys():
            if abs(abs(val2)-root)<10**-5:
                if val2>=0:
                    sign=""
                else:
                    sign="-"
                return sign+rootList[root]
        return str(np.round(val2,4))

    if isinstance(val, complex):
        if val.real!=0:
            term1 = internal(val.real)
        else:
            term1=""
        if val.imag!=0:
            term2 = internal(val.imag)+"j"
            if term1!="":
                term2 =" + "+ term2
        else:
            term2=""
        if term1=="" and term2=="":
            term1="0"

        return term1+term2

    else:
        return internal(val)