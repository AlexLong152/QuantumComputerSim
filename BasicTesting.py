# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy
import QuantComp as qc
from importlib import reload as re
re(qc)

def testBasis1():
    basis=qc.oneZeroBasis()
    print("For one, zero basis")
    print("vec1 is", basis.vec1)
    print("vec2 is", basis.vec2)

    print("vec1Str is", basis.vec1Str)
    print("vec2Str is", basis.vec2Str)
    print(basis)

def testBasis2():
    basis=qc.pmBasis()
    print("For plus minus basis")
    print("vec1 is", basis.vec1)
    print("vec2 is", basis.vec2)

    print("vec1Str is", basis.vec1Str)
    print("vec2Str is", basis.vec2Str)
    print(basis)

def testStr():
    a =np.array([1,2,3])
    print(qc.arr2SqrtStr(a))
    b= np.array([-np.sqrt(1/2),1+-1j*np.sqrt(3)/2],dtype=np.complex128)
    print(qc.arr2SqrtStr(b))

    basis=qc.pmBasis()
    print(basis)

def testOps():
    # print(qc.pauliX)
    # print(qc.pauliY)
    # print(qc.pauliZ)
    # print(qc.H)
    ang = np.pi/6
    print(qc.T(ang))
    print(qc.R(ang))
    print(qc.K(ang))

if __name__=="__main__":
    # testBasis1()
    # testBasis2()
    # testStr()
    testOps()