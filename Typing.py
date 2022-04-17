# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy, copy


class basis:
    """
    vec1 and vec2 are represented in the standard |0>, |1> basis
    """

    def __init__(self, vecs, vec1Str, vec2Str):
        self.vecs=np.array(vecs)
        self.vec1Str = vec1Str
        # self.vec1Repr = vec1Repr

        self.vec2Str = vec2Str
        # self.vec2Repr = vec2Repr
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

def arr2SqrtStr(vec):
    """
    Represents some square roots of values as strings that include sqrt(stuff)
    TODO: extend this to complex values for vec

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

class oneZeroBasis(basis):
    def __init__(self):
        vec1=np.array([1, 0],dtype=np.complex128)
        # print(type(vec1[0]))
        vec2=np.array([0, 1],dtype=np.complex128)
        super().__init__(np.array([vec1, vec2]), "0", "1")

class pmBasis(basis):
    def __init__(self):
        vec1=1/np.sqrt(2)*np.array([1, 1],dtype=np.complex128)
        # print(type(vec1[0]))
        vec2=1/np.sqrt(2)*np.array([1, -1],dtype=np.complex128)
        super().__init__(np.array([vec1, vec2]), "0", "1")

def operString2Mat(stringOper, basis):
    pass


def changeBasis():
    pass


class oper:
    """
    Operator class
    """
    def __init__(self, mat, basis=oneZeroBasis()):
        if isinstance(mat, np.ndarray):
            self.mat = deepcopy(mat).astype(np.complex128)  # This is the current representation

        elif isinstance(mat, str):
            self.mat = operString2Mat(mat, basis)
        else:
            raise TypeError("oper type not supported")

        self.basis = basis  # basis

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.oper = changeBasis(self.oper, newBasis)
            self.basis = newBasis

    def __str__(self, whichBasis=False):
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
        return strMat

pauliX = np.array([[0,1],
                  [1,0]],dtype=np.complex128)
pauliX = oper(pauliX)

pauliY = 1j*np.array([[0,-1],
                     [1,0]],dtype=np.complex128)
pauliY = oper(pauliY)

pauliZ = np.array([[1,0],
                  [0,-1]],dtype=np.complex128)
pauliZ = oper(pauliZ)

H = 0.5**(1/2)* np.array([[1,1],
                          [1,-1]],dtype=np.complex128)
H = oper(H)

def T(alpha):
    """
    T operator, Degrees are in radians
    """
    t1 = np.e**(1j*alpha)
    t2 = np.e**(-1j*alpha)
    mat = np.array([[t1,0],
                    [0,t2]],dtype=np.complex128)
    return oper(mat)

def R(beta):
    """
    R operator, Degrees are in radians
    """
    cosBeta = np.cos(beta)
    sinBeta = np.sin(beta)
    mat = np.array([[cosBeta,sinBeta],
                    [-sinBeta,cosBeta]],dtype=np.complex128)
    return oper(mat)

def K(delta):
    """
    K operator, Degrees are in radians
    """
    t1 = np.e**(1j*delta)
    mat = np.array([[t1,0],
                    [0,t1]],dtype=np.complex128)
    return oper(mat)