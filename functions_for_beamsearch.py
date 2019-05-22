# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random as rand
import copy

# initialise a random sequence
shape=(1,30)
def genSequence():
    sequence=np.zeros(shape=shape,dtype=int)
    sequence[0,0]=1
    for i in range(1,30):
        r=rand.randint(0,100)
        endCaption=rand.random()
        if endCaption>0.9:
            sequence[0,i]=2
#            print('sequence ended at element',i)
            return sequence
        sequence[0,i]=r
        if r==2:
#            print('sequence ended at element',i)
            return sequence
    return sequence
def genCaption():
    sequence=genSequence()
    confidence=0
    caption={'sequence':sequence,'confidence':0}
    for i in range(30):
        if sequence[0,i]!=0:
            r=rand.random()
            confidence=confidence+r
    caption={'sequence':sequence,'confidence':confidence}
    return caption

def getAvgConfidence(caption):
    sequence=caption['sequence']
    confidence=caption['confidence']
#    get the length of the sequence
    l=getSeqLen(sequence)
    avgConfidence=confidence/l
    return avgConfidence

def getSeqLen(sequence,verbose=0):
    l=0
    for i in range(30):
        if sequence[0,i]!=2:
            l+=1
        else:
            l+=1
            break
    if verbose:
        print(l)
    return l

def nth_best(vector,n):
    v=copy.copy(vector)
#    discard (n-1) biggest elements
    for i in range(n-1):
        best=np.argmax(v)
        v[best]=-100
    best_position=np.argmax(v)
    best_value=max(v)
    return best_position,best_value