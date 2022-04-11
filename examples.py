# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy
import QuantComp as qc

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
    print(qc.myStr(a))
    b= np.array([-np.sqrt(1/2),-np.sqrt(3)/2])
    print(qc.myStr(b))

    basis=qc.pmBasis()
    print(basis)

if __name__=="__main__":
    # testBasis1()
    # testBasis2()
    testStr()