#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import  pickle
import  numpy as np
import matplotlib.pyplot as plt
from  decisiontree  import  DecisionTree

# data : tableau (films ,features), id2titles : dictionnaire  id -> titre ,
# fields : id  feature  -> nom
[data , id2titles , fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la  derniere  colonne  est le vote
datax=data [: ,:32]
datay=np.array ([1 if x[33] >6.5  else  -1 for x in data])

tot = len(data)
profondeurs = range(1, 15, 2)

def scoreTrain():
    scores = []
    for depth in profondeurs:
        dt = DecisionTree(depth)
        dt.fit(datax ,datay)
        #dt.predict(datax [:5 ,:])
        scores.append(dt.score(datax ,datay))
        # dessine l’arbre  dans un  fichier  pdf   si pydot  est  installe.
        #dt.to_pdf("/tmp/test_tree.pdf",fields)
        # sinon  utiliser  http :// www.webgraphviz.com/
        #print(dt.to_dot(fields))
        #ou dans la  console
        #print(dt.print_tree(fields ))
    return scores

def scoreTrainTest(f:float):
    assert(f>0 and f<=1)
    l = int(tot*f)
    scoresTrain = []
    scoresTest = []
    for depth in profondeurs:
        dt = DecisionTree(depth)
        dt.fit(datax[:l] ,datay[:l])
        scoresTrain.append(dt.score(datax[:l] ,datay[:l]))
        scoresTest.append(dt.score(datax[l:] ,datay[l:]))
    return scoresTrain, scoresTest

def scoreCross(n=5):
    """fait la moyenne sur n tests
    taille test = tot/n"""
    assert(type(n) == int)
    scoresTrain = []
    scoresTest = []
    for depth in profondeurs:
        sTrain=0
        sTest=0
        for i in range(n):
            start = tot*i//n
            end = tot*(i+1)//n
            dt = DecisionTree(depth)
            xtrain = np.vstack((datax[:start],datax[end:]))
            ytrain = np.hstack((datay[:start],datay[end:]))
            dt.fit(xtrain ,ytrain)
            sTrain += dt.score(xtrain ,ytrain)
            sTest += dt.score(datax[start:end] ,datay[start:end])
        scoresTrain.append(sTrain/n)
        scoresTest.append(sTest/n)
    return scoresTrain, scoresTest

#plt.plot(r, scoreTrain())
for f in [.8, .5, .2]:
    train, test = scoreTrainTest(f)
    plt.plot(profondeurs, train)
    plt.plot(profondeurs, test)
    plt.title("trainTest"+str(f))
    plt.show()
 
for n in [2, 5, 10]:
    train, test = scoreCross(n)
    plt.plot(profondeurs, train)
    plt.plot(profondeurs, test)
    plt.title("cross"+str(n))
    plt.show()
