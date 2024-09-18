#%matplotlib inline
import matplotlib.pyplot as plt
import quandl
import datetime
from datetime import date
import pandas as pd
import numpy as np
import csv
#quandl.ApiConfig.api_key = "63gdVnc_-LzW9XyB1Ajk"

class Optimizer():

    def __init__(self, stocksymlist):
        self.stocksymlist = stocksymlist
        self.dfDict = {}
        self.fillDict()
        self.sigmas = []
        self.drifts = []
        self.closes = []
        self.cgrs = []
        self.covmat = []
        self.calcCGRs()
        self.target_rate = np.max(self.drifts)+abs(np.min(self.drifts))
        safe_weights = np.array([1.0/len(self.sigmas) for i in range(len(self.sigmas))])*4

        self.target_risk = self.risk(safe_weights)/4


    def fillDict(self):
        for sym in self.stocksymlist:
            df_close = pd.DataFrame()
            df_temp = pd.read_json('https://api.iextrading.com/1.0/stock/'
                +sym+'/chart/5y')
            df_temp.set_index('date',inplace=True)
            #df_close = df_temp[['volume','close']]
            df_close = df_temp[['close']]

            self.dfDict[sym] = df_close


    def calcCGRs(self):
        keys = self.dfDict.keys()
        self.sigmas = []
        self.drifts = []
        self.closes = []
        self.cgrs = []
        for key in keys:
            stock = self.dfDict[key]
            price = np.array(stock.close)
            pricep1 = np.roll(price,1)
            lnratio = price/pricep1
            cgr = np.log(lnratio)
            cgr[0]=999.99
            stock['cgr'] = cgr
            self.closes.append(price[1:])
            sigma = np.std(cgr[:-252])
            self.sigmas.append(sigma)
            drift = np.mean(cgr[:-252])
            self.drifts.append(drift)
            self.cgrs.append(np.array(cgr[1:]))
        self.closes = np.array(self.closes)
        self.sigmas = np.array(self.sigmas)
        self.drifts = np.array(self.drifts)
        self.covmat = np.cov(self.cgrs)
        self.cgrs = np.array(self.cgrs)


    def risk(self, weights):
        cv = np.dot(self.covmat, weights)
        return (np.dot(weights, cv))**0.5
        # sig_sq = float(0)
        # for i in range(len(weights)):
        #     for j in range(len(weights)):
        #         sig_sq += weights[i]*weights[j]*self.covmat[i][j]
        # return sig_sq**0.5


    def return_rate(self, weights):
        return np.dot(weights, self.drifts)
        # rate = float(0)
        # for i in range(len(weights)):
        #     rate += weights[i]*self.drifts[i]
        # return rate


    def KL(self, p_rate, p_sig, q_rate, q_sig):
        a = np.log(q_sig/p_sig)
        b = (p_sig**2 + (p_rate-q_rate)**2)/(2*q_sig**2)
        return a + b - 0.5


    def weighted_cov_sum(self, i, weights):
        wsum = float(0)
        for j in range(len(weights)):
            if j==i:
                wsum += 2*weights[i]*self.covmat[i][i]
            else:
                wsum += weights[j]*self.covmat[i][j]
        return wsum


    def gradient(self, weights):
        sig1 = risk(weights)
        mu1 = return_rate(weights)
        grad = np.zeros(len(weights))
        for i in range(len(grad)):
            wsum = self.weighted_cov_sum(i, weights)
            grad[i] = (wsum/(2*(target_risk**2))
                ) + ((self.drifts[i]*mu1)/(target_risk**2)
                ) - (wsum/(2*(sig1**2)))
        return grad


    def kl_sharpe_descent(self, epochs, learn_rate):
        #initialize and normalize weights
        #weights = np.random.normal(loc=0.5, scale=0.1, size=len(self.stocksymlist))
        weights = np.zeros(len(self.stocksymlist))+float(1)
        weights = weights/np.sum(weights)
        last_error = float("inf")
        for epoch in range(epochs):
            
            #normalize the weights
            if np.min(weights)<0:
                weights = weights - np.min(weights)
            weights = weights/np.sum(weights)
            
            #take stddev and mean of predictions
            pred_sig = risk(weights)
            pred_rate = return_rate(weights)
            
            #compare to target using KL div
            error = self.KL(pred_rate, pred_sig, 
                self.target_rate, self.target_risk)
            
            #get gradient for these weights
            grad = self.gradient(weights)
            
            
            #update weights
            weights += error*grad*learn_rate
            
            #change learning rate
            if error/last_error > 1.04:
                learn_rate *= 0.9
            elif error < last_error:
                learn_rate *= 1.05
            
            last_error = error
                
            #         if epoch%200 == 0:
            #             print("loss", error)
            
        if np.min(weights)<0:
            weights = weights-np.min(weights)
        weights = weights/np.sum(weights)

        return (weights, error)


    def sharpe(self, weights):
        return self.return_rate(weights)/self.risk(weights)


    def sharpe_grad(self, weights):
        risk = self.risk(weights)
        variance = risk**2
        cpi = np.dot(self.covmat, weights)
        grad = np.zeros(len(weights))
        for i in range(len(weights)):
            dpi = np.zeros(len(weights))
            dpi[i] = float(1)
            dsig = (np.dot(dpi,cpi)+np.dot(weights, np.dot(self.covmat, dpi))
                )/(2*risk)
            numer = weights[i]*self.drifts[i]*risk - np.dot(weights,
                self.drifts)*dsig
            grad[i] = numer/variance
        return grad



    def ascent(self, epochs, learn_rate, short=True):
        weights = np.zeros(len(self.stocksymlist)-1)+float(1)
        weights = weights/np.sum(weights)
        last_sharpe = -float("inf")
        for epoch in range(epochs):
            
           
            
            fullweights = [i for i in weights]
            fullweights.append(float(1-np.sum(weights)))
            fullweights = np.array(fullweights)

             #get gradient for these weights
            grad = self.sharpe_grad(fullweights)

            #update weights
            weights += grad[0:-1]*learn_rate
            
            fullweights = [i for i in weights]
            fullweights.append(float(1-np.sum(weights)))
            fullweights = np.array(fullweights)
            
            #calculate new sharpe
            sharpe = self.sharpe(fullweights)

            #change learning rate
            if sharpe/last_sharpe < 0.99:
                learn_rate *= 0.8
            elif sharpe/last_sharpe > 1.02:
                learn_rate *= 1.05
            
            last_sharpe = sharpe
                
            if epoch%100 == 0:
                print("sharpe", sharpe)
                print(weights)

        return (weights, sharpe)












