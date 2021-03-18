#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:59:36 2020

@author: thibautmartin
"""
import numpy as np
import matplotlib.pyplot as plt


def featureNormalize(X):
    Xs = (X-X.mean(axis =0))/ X.std(axis=0)
    
    return Xs 
    
def initialisation(A):
    B = np.ones((A.shape[0],1))
    C = np.concatenate((B,A),axis = 1)

    return C


def normalization(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        mu[0,i]= X[:,i].mean()
        sigma[0,i]=X[:,i].std()
    for i in range(X.shape[1]):
        X_norm[:,i]= (X[:,i] - mu[:,i])/sigma[:,i]
    
    return [X_norm, mu, sigma]
    


def initialisation_(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        mu[1,i]= X[:,i].mean()
        sigma[1,i]=X[:,i].std
    for i in range(X.shape[1]):
        X_norm[:,i]=X 
    
    return [X_norm, mu, sigma]

def computeCost(X,y, theta):
    m = y.shape[0]
    J = 0
    predictions = X.dot(theta)
    sqrErr = (predictions - y)**2
    J = 1 /(2*m) *sum(sqrErr)
   
    return J

def linear_regression(X_train, Y_train, X_test,Y_test, learning_rate, num_iters, print_cost=True):
    
    #add intercept term to X
    X_train = initialisation(X_train)
    X_test = initialisation(X_test)
    X = np.concatenate((X_train, X_test), axis = 0)
    Y = np.concatenate((Y_train, Y_test), axis = 0)
    
    
    train_costs = []
    test_costs = []
    
    theta = np.zeros((X.shape[1],1))
    m= Y_train.shape[0]
    
    for i in range(0,num_iters):
        theta = theta - (learning_rate/m) * (X_train.T).dot(X_train.dot(theta) - Y_train)
        train_cost= computeCost(X_train,Y_train,theta)
        train_costs.append(train_cost)
        test_cost = computeCost(X_test,Y_test, theta)
        test_costs.append(test_cost)
        
        if print_cost and i % 100 == 0:
            print("train_cost after iteration %i: %f" %(i, train_cost))
            print("test_cost after iteration %i: %f" %(i, test_cost))
   
    
    # plot the costs
    plt.figure(figsize = (16,9))
    plt.style.use('seaborn')
    plt.plot(np.squeeze(train_costs))
    plt.ylabel('cost')
    plt.plot(np.squeeze(test_costs))
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.legend(['train_cost', 'test_cost'])

    plt.show()
    
    #evaluate
    eval_dict = evaluate(X_train, Y_train, X_test, Y_test, theta, print_eval= True)
    
    return {'theta': theta, 'train_costs': train_costs, 'test_costs': test_costs}, eval_dict


def predict(X,theta, init = False):
    # Add bias term and predict 
    # Return a normalized prediction
    if init==True:
        X = initialisation(X)
    return X.dot(theta)


def evaluate(X_train, Y_train, X_test, Y_test, parameters, print_eval = True):
   
    X = np.concatenate((X_train, X_test), axis = 0)
    Y = np.concatenate((Y_train, Y_test), axis = 0)
    
    results = predict(X, parameters)
    train_results = predict(X_train, parameters)
    test_results = predict(X_test, parameters )
    
    # Mean absolute error 
    MAE = np.sum(np.absolute(Y-results))/ Y.shape[0]
    train_MAE = np.sum(np.absolute(Y_train-train_results))/ Y_train.shape[0]
    test_MAE = np.sum(np.absolute(Y_test-test_results))/ Y_test.shape[0]
    
    # Mean Square Error 
    MSE = np.sum((Y-results)**2)/ Y.shape[0]
    train_MSE = np.sum((Y_train-train_results)**2)/ Y_train.shape[0]
    test_MSE = np.sum((Y_test-test_results)**2)/ Y_test.shape[0]
    
    #RMSE
    RMSE = np.sqrt(MSE)
    train_RMSE = np.sqrt(RMSE)
    test_RMSE = np.sqrt(RMSE)
    
    #Mean
    MEAN = np.mean(Y)
    
    
    #R2 
    R2 = 1- np.sum((Y-results)**2)/np.sum((Y-np.mean(Y))**2)
    
    
    eval_dict = {'MAE': MAE,
               'train_MAE': train_MAE,
               'test_MAE': test_MAE,
                'MSE': MSE,
                'train_MSE': train_MSE,
                'test_MSE': test_MSE,
                'RMSE': RMSE,
                'train_RMSE': train_RMSE,
                'test_RMSE': test_RMSE,
                'MEAN': MEAN,
                'R2': R2}
    
    for k, v in eval_dict.items():
        eval_dict[k] = round(v, 3)
    
    if print_eval:
        for k, v in eval_dict.items():
            print(f"{k:<15}{v:>15}")
    
    return eval_dict

    
    
 