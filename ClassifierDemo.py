# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:11:39 2018

@author: Ankit_Kumar34
"""
from sklearn import tree
#height weight and shoe size
X=[[181,80,44],[170,70,43],[160,60,38],[166,65,40]]
Y=['male','female','female','male']
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,Y)
pridiction=clf.predict([[150,30,30]])
print (pridiction)