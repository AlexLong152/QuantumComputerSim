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
            out+=myStr(vec)+"\n"
        return out+"}"

def myStr(vec):
    rootList = {1/np.sqrt(2):"1/sqrt(2)",
        np.sqrt(3)/2:"sqrt(3)/2"}
    out = "< "
    for val in vec:
        setVal=False
        for root in rootList.keys():
            if abs(abs(val)-root)<10**-5:
                if val>=0:
                    sign=""
                else:
                    sign="-"
                out+=sign+rootList[root]+","
                setVal=True
        if not setVal:
            out+=str(val)+","
    return out[:-1]+" >"

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


class operator:

    def __init__(self, oper, basis=oneZeroBasis()):
        if isinstance(oper, np.ndarray):
            self.oper = deepcopy(oper)  # This is the current representation

        elif isinstance(oper, str):
            self.oper = operString2Mat(oper, basis)
        else:
            raise TypeError("oper type not supported")

        self.basis = basis  # basis

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.oper = changeBasis(self.oper, newBasis)
            self.basis = newBasis


# def isSqrt(val):
#     """
#     Tests if val is square root of an integer
#     """
#     print("val=",val)
#     valSquare=val*val
#     intValSquare=int(valSquare)
#     print("int(valSquaure)=",intValSquare)
#     print("val**2=",valSquare)
#     print("valSquare-int(valSquare)=",valSquare-int(valSquare))
#     print("\n\n")
#     if abs(valSquare-int(valSquare))<10**-5:
#         return True