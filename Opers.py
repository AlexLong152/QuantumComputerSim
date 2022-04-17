# -*- coding: utf-8 -*-

"""
@author: alexl

This file is for the definitions of commonly used operators
"""
import numpy as np
import Typing as tp
# from copy import deepcopy, copy

pauliX = np.array([[0,1],
                  [1,0]],dtype=np.complex128)
pauliX = tp.oper(pauliX)

pauliY = 1j*np.array([[0,-1],
                     [1,0]],dtype=np.complex128)
pauliY = tp.oper(pauliY)

pauliZ = np.array([[1,0],
                  [0,-1]],dtype=np.complex128)
pauliZ = tp.oper(pauliZ)

H = 0.5**(1/2)* np.array([[1,1],
                          [1,-1]],dtype=np.complex128)
H = tp.oper(H)

def T(alpha):
    """
    T operator, Degrees are in radians
    """
    t1 = np.e**(1j*alpha)
    t2 = np.e**(-1j*alpha)
    mat = np.array([[t1,0],
                    [0,t2]],dtype=np.complex128)
    return tp.oper(mat)

def R(beta):
    """
    R operator, Degrees are in radians
    """
    cosBeta = np.cos(beta)
    sinBeta = np.sin(beta)
    mat = np.array([[cosBeta,sinBeta],
                    [-sinBeta,cosBeta]],dtype=np.complex128)
    return tp.oper(mat)

def K(delta):
    """
    K operator, Degrees are in radians
    """
    t1 = np.e**(1j*delta)
    mat = np.array([[t1,0],
                    [0,t1]],dtype=np.complex128)
    return tp.oper(mat)