#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Bayes algorithm

# calculate P(y)
def calcPrior(df, y_name):
    # count valued y entries to total occurences
    y_labels = df[y_name].unique()
    prior = df[y_name].value_counts(normalize=True).sort_index()
    return prior.index.tolist(), prior.tolist()

# convert a column of continuous values to discrete
def convert_Continuous(df, x_name, mean):
    df.loc[(df[x_name] <= mean), x_name] = False
    df.loc[(df[x_name] > mean), x_name] = True
    return "mean", mean

# calculate P(xn | y) for all entries in column xn
def calcLikelihood_Discrete(df, x_name, y_name):
    likelihood = df.groupby([y_name])[x_name].value_counts(normalize=True).sort_index(level=[y_name, x_name])
    return list(set(list(zip(*likelihood.index.tolist()))[1])), likelihood.tolist()

def naiveBayes(df, x_names, y_name, x_typecont):    
    # 1. find P(y):
    #    fill y_labels, y_values in calcPrior(), get the results back
    y_labels, y_values = calcPrior(df, y_name)
    
    # 2. find P(x|y) for each x in xn
    x_labels = pd.DataFrame()
    x_values = pd.DataFrame()

    for x_name, cont in zip(x_names, x_typecont):
        if cont:
            mean = df[x_name].mean()
            mean_label, mean_value = convert_Continuous(df, x_name, mean)
        new_label, new_value = calcLikelihood_Discrete(df, x_name, y_name)
        if cont:
            new_label.append(mean_label)
            new_value.append(mean_value)
        x_labels = pd.concat([x_labels.reset_index(drop=True), pd.DataFrame({x_name: new_label}).reset_index(drop=True)], axis=1)
        x_values = pd.concat([x_values.reset_index(drop=True), pd.DataFrame({x_name: new_value}).reset_index(drop=True)], axis=1)
    
    return y_labels, y_values, x_labels, x_values


def classify(df, x_names, y_name, y_labels, y_values, x_labels, x_values, x_typecont):
    predictions = []
    
    #convert continous columns first
    for x_name, cont in zip(x_names,x_typecont):
        if cont:
            # get the mean from trained parameters
            index = (x_labels[x_name].count()-1)*len(y_labels)
            mean = x_values[x_name][index]
            # convert column with this mean
            convert_Continuous(df, x_name, mean)
    
    # compute posterior probabilities for each y case aka [y=true, y=false]
    for e in range(len(df)):
        posterior = []

        for y_index in range(len(y_labels)):
            # probability = P(y)
            probability = y_values[y_index]

            for x_name, cont in zip(x_names,x_typecont):
                #case 1: entry1, false, false
                # probability = P(y) P(x1|y) P(x2|y)... P(xn|y)
                testval = df[x_name][e]
                x_len = x_labels[x_name].count()
                if cont:
                    x_len-=1
                x_index = x_labels.loc[x_labels[x_name] == testval].index[0]
                prob_x_giv_y = x_values[x_name][y_index*x_len+x_index]
                probability *= prob_x_giv_y

            posterior.append(probability)
            
        predictions.append(y_labels[np.argmax(posterior)])
    return pd.DataFrame(predictions, columns=[y_name])
