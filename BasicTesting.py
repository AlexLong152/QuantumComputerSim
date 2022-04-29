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


def testPsiPrint():
    # this isn't normalized but thats fine for demo purposes
    vec = np.array([1/np.sqrt(2), 1/np.sqrt(2), 1j, -2, 5j])
    psi3 = lib.psi(vec)
    print(psi3)


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
    psi = lib.psi(np.array([1, 0]))
    print("Before change  of basis psi is", psi)
    psi.changeRep(lib.pmBasis)
    print("After change  of basis psi is", psi)

    # print(oper.__str__(whichBasis=True))
    # print("\nBefore change of basis oper is\n", oper)
    # oper.changeRep(lib.pmBasis)
    # print("After change of basis oper is\n", oper)


def testTensorPsi():
    psi1 = lib.psi(np.array([1, 1]))
    psi2 = lib.psi(np.array([1, 1]))
    psi3 = lib.psi(1/np.sqrt(2)*np.array([1, 1j]))

    psi = lib.psiTensorProd(psi1, psi2, psi3)
    psi = lib.psiTensorProd([psi1, psi2, psi3])
    psi2 = deepcopy(psi)
    print("print(psi) before tensor product gives: \n", psi)
    psi.preformProd([0, 1])
    print("print(psi) after tensor product gives: \n", psi)
    print("psi before prod is")
    print(psi)
    psi.preformProd()
    print("doing all the tensor products gives")
    print(psi)
    print("we can also do this all at once and should get the same answer")
    psi2.preformProd()
    print(psi2)
    print("\n\n\n")


def testNumBasis():
    psi = lib.psi(np.array([1, 1, 1, 1]))
    print(psi)
    psi.normalize()
    print(psi)


def testTensorMat():
    a = op.pauliX
    b = op.pauliY
    c = op.pauliZ

    o2 = lib.operTensorProd(a, b, c)
    ind = np.array([0, 1])
    o2.preformProd(ind)
    print(o2)


def testTensorPsi2():
    psi1 = lib.psi(np.array([1, 1j]))
    psi2 = lib.psi(np.array([1j, -2]))
    psi3 = lib.psi(1/np.sqrt(2)*np.array([1, 1j]))
    psi4 = lib.psi(np.array([2, 1j]))
    #  psi5 = lib.psi(np.array([3j, 1]))

    # psiTot = lib.psiTensorProd(psi1,psi2,psi2,psi3,psi4)
    psiTot = lib.psiTensorProd(psi1, psi2, psi2, psi3, psi4)
    psiTot2 = deepcopy(psiTot)

    print(psiTot)
    ind = np.array([[0, 1], [2, 3]])
    psiTot.preformProd(ind)
    print(psiTot)

    psiTot.preformProd()
    psiTot2.preformProd()

    print("psiTot:\n", psiTot)
    print("psiTot2:\n", psiTot2)
    # print("Compare arrays vy checking for zeros:", (psiTot.vec-psiTot2.vec==0).all())
    # print("Equality check yields", psiTot==psiTot2)


def testToSqrt():
    val = 1 - (1/np.sqrt(2))*1j
    ar = np.array([val, -1, -1j, 1])
    psi = lib.psi(ar)
    # print(lib.toSqrtStr(val))
    print(psi)


def testOperOnPsi():
    psi1 = lib.psi(np.array([1, -1]))

    psi2 = lib.psi(np.array([2, 2j]))

    opr1 = op.pauliX
    opr2 = op.pauliY
    psi = lib.psiTensorProd(psi1, psi2)
    psi.normalize()
    opr = lib.operTensorProd(opr1, opr2)
    psiOut = lib.operOnPsi(opr, psi)
    print(psiOut)


def testOpr1():
    opr1 = op.pauliX
    psi1 = lib.psi(np.array([1, 0]))
    psi2 = lib.operOnPsi(opr1, psi1)
    print(psi2)


def testMeasure():
    # psi1 = lib.psi(np.array([1, 1j]))
    # psi2 = lib.psi(np.array([1j, -2]))
    # psiTot = lib.psiTensorProd(psi1,psi2,)

    psiTot = lib.psi(1/np.sqrt(2)*np.array([1, 1j]))
    psiTot.normalize()
    # print(psiTot)
    ar = lib.measure(psiTot)
    print("Probabilities are:", ar)
    ind = lib.randMeasure(psiTot)
    print("randomly choisen ind = ", ind)


if __name__ == "__main__":
    # testBasis1()
    # testBasis2()
    # testStr()
    # testOps()
    # changeBasis()
    # testNumBasis()
    # testTensorPsi()
    # testTensorMat()
    # testTensorPsi2()
    # testToSqrt()
    testOperOnPsi()
    # testPsiPrint()
    # testMeasure()
    # print(op.pauliX)
    # testOpr1()
