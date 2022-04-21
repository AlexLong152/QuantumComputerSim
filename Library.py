# -*- coding: utf-8 -*-

"""
@author: alexl
"""
import numpy as np
from copy import deepcopy, copy
from collections.abc import Iterable
import Library as lib # importing self for type checking

class basis:
    """
    vec1 and vec2 are represented in the standard |0>, |1> basis
    Atributes
    -------------
    vecs: 2d ndarray
        vecs[i] gives the ith vector

    vecStings: 1d ndarray of strings
        vecStrings[i] should give the representation of the ith vector as it should be represented in the ket
    """

    def __init__(self, vecs, vecStrings):
        # self.vecs = np.array(vecs)
        self.vecs = vecs
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


def operMakeNumBasis(opr):
    """
    makes basis |0>, |1>, |2>... for operator of dimension n
    """
    nums = np.arange(len(opr.mat[0]),dtype=int)
    strings = np.array([str(i) for i in nums])
    vecs = np.identity(len(opr.mat[0]))
    return basis(vecs, strings)

def makeNumBasis(wvPsi):
    """
    makes basis |0>, |1>, |2>... for psi of dimension n
    """
    nums = np.arange(len(wvPsi.vec),dtype=int)
    strings = np.array([str(i) for i in nums])
    vecs = np.identity(len(wvPsi.vec))
    return basis(vecs, strings)

_vec1 = 1/np.sqrt(2)*np.array([1, 1], dtype=np.complex128)
_vec2 = 1/np.sqrt(2)*np.array([1, -1], dtype=np.complex128)
_strings = np.array(["+", "-"])
pmBasis = basis(np.array([_vec1, _vec2]), _strings)


_vec1 = np.array([1, 0], dtype=np.complex128)
# print(type(vec1[0]))
_vec2 = np.array([0, 1], dtype=np.complex128)
_strings = np.array(["0", "1"])
oneZeroBasis = basis(np.array([_vec1, _vec2]), _strings)

def operOnPsi(opr, wvPsi):
    """
    Preforms the operator operation on the wavefunction psi
    op, and wvPsi can be their normal version, or their tensor version, but if they are the tensor version 
    then the dimensions have to match
    """
    opr =copy(opr)
    wvPsi =copy(wvPsi)

    if type(wvPsi) == lib.psi:
        if opr.basis!= wvPsi.basis:
            numBasis = makeNumBasis(wvPsi)
            opr.changeRep(numBasis)
            wvPsi.changeRep(numBasis)            

        vec = np.dot(opr.mat,wvPsi.vec)
        newPsi = psi(vec,basis=wvPsi.basis)
        return newPsi

    else:
        assert type(wvPsi) == lib.psiTensorProd
        n = len(wvPsi.psis)
        psiOut = np.empty(n,dtype=object)
        for i in range(len(wvPsi.psis)):
            psiTmp = wvPsi.psis[i]
            oprTmp = opr.mats[i]

            if oprTmp.basis!= psiTmp.basis:
                numBasis = makeNumBasis(psiTmp)
                oprTmp.changeRep(numBasis)
                psiTmp.changeRep(numBasis)            

            psiOut[i] = operOnPsi(oprTmp,psiTmp) # recursive lol

        psiOut = psiTensorProd(psiOut)
        return psiOut

class psi:
    """
    The Wavefunction

    if no basis is passed, then assumes the representation is just the standard numeric basis

    Attributes
    -----------
    vec: 1d ndarray
        The coefficents of the basis
    basis: optional, instance of basis class
        Which basis the vecs belong to
    
    normalize: function
        void function that normalizes vectors
    """

    def __init__(self, vec, basis=None):

        if isinstance(vec, np.ndarray):
            # This is the current representation
            self.vec = deepcopy(vec).astype(np.complex128)

        if isinstance(basis, type(None)):
            self.basis = makeNumBasis(self)
        else: 
            self.basis = basis

    def changeRep(self, newBasis):
        if newBasis != self.basis:
            self.vec = waveChangeBasis(self.vec, self.basis, newBasis)
            self.basis = newBasis

    def normalize(self, returnC=False):
        squared = np.conjugate(self.vec)*self.vec
        c = np.sqrt(np.sum(squared).real)
        self.vec = self.vec/c
        if returnC:
            return c, self

    def __str__(self):
        out = ""
        for i, x in enumerate(copy(self.vec)):
            realStr =toSqrtStr(x.real)
            imagStr = ""
            if x.imag!=0:
                imagStr = toSqrtStr(x.imag)+"j"
            else:
                imagStr = ""

            if imagStr == "0j" or imagStr=="0.0j":
                imagStr = ""

            # if i==0 and signStr(x.real)=="+":
            #     realStr = toSqrtStr(x.real)
            if i!=0:
                preString = " "
            else: 
                preString = ""
            if realStr=="0" and imagStr!=0:
                realStr=""
            if i!=0 and x.real>0:
                realStr = "+ "+ realStr
            if i!=0 and realStr == "" and x.imag>0:
                imagStr = "+"+imagStr
            out += preString+realStr+imagStr+"|"+self.basis.vecStrings[i]+"> "
        return out

def signStr(val):
    if val>=0:
        return "+"
    if val<0:
        return "-"
       
class psiTensorProd:
    """
    A class for a wavefunction that is a tensor prodcut of other states.
    Using *psi allows for any number of psi values to be passed, for example

    psi1 = lib.psi(np.array([1, 0]))
    psi2 = lib.psi(np.array([0, 1]))
    psi3 = lib.psi( 1/np.sqrt(2)*np.array([1, 1j]))

    psi = lib.psiTensorProd(psi1,psi2,psi3)

    can also init with an array, or list and it is handled automatically
    psi = lib.psiTensorProd([psi1,psi2,psi3])
    """
    def __init__(self, *psis):
        if isinstance(psis[0], (np.ndarray, list, tuple)):
            if len(psis)>1:
                raise ValueError("If you are going to pass psis as array, can only have one arg")
            psis = psis[0]

        if len(psis)==1:
            raise ValueError("You should use the regular psi class, not the tensor product version")

        self.psis = np.asarray(psis)

    def preformProd(self, indicies=None):
        """
        Preforms the tensor product on the indicies given.
        If all tensor products are preformed, then automatically turns this class into a psi class

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
        if isinstance(indicies, type(None)):
            indicies=np.arange(len(self.psis),dtype=int)

        if len(np.shape(indicies))==1: #deal with case when indicies is just passed as an array
            indicies=np.array([indicies])

        if not isConsecutive(indicies):
            raise ValueError("indicies must be consecutive")

        # flatInd = np.ndarray.flatten(indicies)
        firstVals = copy(indicies[:,0])
        psiList = []
        i=0
        # print("indicies=",indicies)
        while i < len(self.psis):
            if i in firstVals:
                vec = np.array(1, dtype=np.complex128)
                loc = np.where(firstVals==i)[0][0]
                for j in indicies[loc]:
                    psi1 = deepcopy(self.psis[j])
                    psi1.changeRep(makeNumBasis(psi1))
                    vec = np.kron(vec, psi1.vec)

                psiVal = psi(vec) # Basis made automatically
                psiList.append(psiVal)

                i = indicies[loc][-1]+1
            else:
                psiList.append(self.psis[i])
                i+=1
        if len(psiList)==1:
            # if theres only one element left, then don't have to represent this as 
            # a tensor product anymore, so just represent it at a psi class
            self.__class__= psi
            self.__init__(psiList[0].vec)
            # print("type of self is",type(self))
        else:
            self.psis= np.array(psiList)

    def normalize(self):
        Psi2 = deepcopy(self)
        Psi2.preformProd()
        # print(type(Psi2))
        c = Psi2.normalize(returnC=True)

        for i in range(len(self.psis)):
            self.psis[i].vec= self.psis[i].vec/c
        return self

    def __str__(self):
        out = ""
        for psi in self.psis:
            out += "( " + psi.__str__() + " )" + u" \u2297  " #  unicode for tensor product
        out = out[:-3]
        return out
        

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

class operTensorProd:
    """
    A class for representing a tensor product of operators
    can initalize as multiple arguments, or as one list as seen below

    a = opmatList
    b = op.pauliY
    c = op.pauliZ
    lib.operTensorProd([a,b,c])
    lib.operTensorProd(a,b,c)

    args
    -----
    matsList: instance of operator class, or ndarray or list of operator class

    *othermats: other instances of operator class
    """

    def __init__(self, mats, *otherMats):
        if len(otherMats)==0:
            self.mats = mats
        else:
            tmp = []
            tmp.append(mats)
            for val in otherMats:
                tmp.append(val)

            self.mats = np.array(tmp, dtype=object)

        # [print(mat) for mat in self.mats]

    def preformProd(self, indicies=None):
        """
        Preforms the tensor product on the indicies given.
        If all tensor products are preformed, then automatically turns this class into a psi class

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
        if isinstance(indicies, type(None)):
            indicies=np.arange(len(self.mats),dtype=int)

        if len(np.shape(indicies))==1: #deal with case when indicies is just passed as an array
            indicies=np.array([indicies])

        if not isConsecutive(indicies):
            raise ValueError("indicies must be consecutive")

        # flatInd = np.ndarray.flatten(indicies)
        firstVals = copy(indicies[:,0])
        matList = []
        i=0
        # print("indicies=",indicies)
        while i < len(self.mats):
            if i in firstVals:
                mat = np.array(1, dtype=np.complex128)
                loc = np.where(firstVals==i)[0][0]
                # print("j=", j)
                # print("ind[i]=", indicies[i])
                for j in indicies[loc]:
                    mat1 = deepcopy(self.mats[j])
                    # print("mat to be taking prodcut with is\n", mat1)
                    mat1.changeRep(operMakeNumBasis(mat1))
                    mat = np.kron(mat, mat1.mat)
                    # print("mat after this step is\n",mat)
                #     print("vec=", vec)
                # print("len(vec)=",len(vec))
                matVal = oper(mat) # Basis made automatically
                matList.append(matVal)

                i = indicies[loc][-1]+1
                # print("i after increment=", i)
                # print("len(self.psis)=",len(self.psis))
            else:
                matList.append(self.mats[i])
                i+=1

        if len(matList)==1:
            # if theres only one element left, then don't have to represent this as 
            # a tensor product anymore, so just represent it at a psi class
            self.__class__= oper
            self.__init__(matList[0].mat)
            # print("type of self is",type(self))
        else:
            self.mats= np.array(matList,dtype=object)


    def __str__(self):
        out = ""
        for mat in self.mats:
            out += mat.__str__() + u" \u2297  " +"\n"#  unicode for tensor product
        out = out[:-6]
        return out

class oper:
    """
    Operator Class
    """

    def __init__(self, mat, basis=None):
        if isinstance(mat, np.ndarray):
            # This is the current representation
            self.mat = deepcopy(mat).astype(np.complex128)

        # elif isinstance(mat, str):
        #     self.mat = operString2Mat(mat, basis)
        else:
            raise TypeError("oper type not supported")

        if isinstance(basis, type(None)):
            self.basis = operMakeNumBasis(self)
        else: 
            self.basis = basis

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
    rootList = {1/np.sqrt(2): "1/sqrt(2)",
                np.sqrt(3)/2: "sqrt(3)/2",
                1.0:"1",
                0.0:'0'}

    term = str(val)
    sign= ""
    for root in rootList.keys():
        if abs(abs(val)-root) < 10**-5:
            term = rootList[root]
            if val >= 0:
                sign = ""
            else:
                sign = "-"

    return sign+term