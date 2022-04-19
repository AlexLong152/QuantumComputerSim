# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy
import Opers as op
import Library as lib
from importlib import reload as re
re(op)
re(lib)


def testBasis1():
    basis = lib.oneZeroBasis
    print("For one, zero basis")
    print("vec1 is", basis.vec1)
    print("vec2 is", basis.vec2)

    print("vec1Str is", basis.vec1Str)
    print("vec2Str is", basis.vec2Str)
    print(basis)


def testBasis2():
    basis = lib.pmBasis
    print("For plus minus basis")
    print("vec1 is", basis.vec1)
    print("vec2 is", basis.vec2)

    print("vec1Str is", basis.vec1Str)
    print("vec2Str is", basis.vec2Str)
    print(basis)


def testStr():
    a = np.array([1, 2, 3])
    print(lib.arr2SqrtStr(a))
    b = np.array([-np.sqrt(1/2), 1+-1j*np.sqrt(3)/2], dtype=np.complex128)
    print(lib.arr2SqrtStr(b))

    basis = lib.pmBasis
    print(basis)


def testOps():
    # print(qc.pauliX)
    # print(qc.pauliY)
    # print(qc.pauliZ)
    # print(qc.H)
    ang = np.pi/6
    print(op.T(ang))
    print(op.R(ang))
    print(op.K(ang))


def changeBasis():
    oper = op.pauliX
    psi = lib.psi(np.array([1, 0]))
    print(oper.__str__(whichBasis=True))
    print("Before change  of basis psi is", psi)
    psi.changeRep(lib.pmBasis)
    print("After change  of basis psi is", psi)

    print("\nBefore change of basis oper is\n", oper)
    oper.changeRep(lib.pmBasis)
    print("After change of basis oper is\n", oper)


def testTensorPsi():
    psi1 = lib.psi(np.array([1, 0]))
    psi2 = lib.psi(np.array([0, 1]))
    psi3 = lib.psi( 1/np.sqrt(2)*np.array([1, 1j]))

    psi = lib.psiTensorProd(psi1,psi2,psi3)
    print(psi)

if __name__ == "__main__":
    # testBasis1()
    # testBasis2()
    # testStr()
    # testOps()
    # changeBasis()
    testTensorPsi()