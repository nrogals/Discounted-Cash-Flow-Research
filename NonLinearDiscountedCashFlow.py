"""
The module here attempts to study the non-linearity found in 
financial market prices. 

The discounted cash-flow can viewed as a linear pricing tool. 

A non-linear discounted cash-flow can be considered, which is what we do here. 

TODO:
    1. Waiting on API from Steve at Value Investing. 


 https://valueinvesting.io/api/dcf?tickers=GOOGL,AAPL&api_key=

Author: 
Nathaniel Rogalskyj 
"""

import numpy as np
import requests
import nasdaqdatalink
import npf
from scipy.linalg import null_space 
import random   

nasdaqdatalink.ApiConfig.api_key = "fxzcvpCP8VkSQsQks51b"


def linearDiscountedCashFlow(w, c):
    w_i = np.array([(1+w)**(-1*(i)) for i in range(len(c))])
    return np.dot(w_i.T, c)


def testLinearDiscountedCashFlow(w,c):
    actual = npf.npv(w, c)
    expected = linearDiscountedCashFlow(w,c)
    assert(actual == expected)

def nonLinearDiscountedCashFlow(w,c):
    w_i = np.array([[(1+w)**(-1*(i)) for i in range(len(c))]])   
    #print("w_i {0} \n".format(w_i))

    ns = null_space(w_i)    
    q,r = np.linalg.qr(w_i)
    angle = np.dot(ns[:,1].T, ns[:,0])
    
    print("Null Space is provided by: {0} for the linear map {1} \n".format(ns, w_i))
    
    assert(np.isclose(angle, 0.0))
    print("Matrix QR {0} and WI {1} \n".format(q*r, w_i))    
    assert(np.isclose(np.linalg.norm(q*r - w_i), 0.0))
    
    projWiAgainstNs = np.matmul(w_i, ns)
    projCAgainstNs = np.matmul(c, ns)        
    wINorm = np.linalg.norm(w_i)
    projCAgainstWi = np.matmul(c, w_i.T / wINorm)      
        
    k = np.matmul(ns, projCAgainstNs[:, np.newaxis])
    r = np.matmul(w_i.T / np.linalg.norm(w_i.T), projCAgainstWi[:,np.newaxis])
    v = k + r
    v = v[:,0]
    print("Norm Lin Alg: \n {0} and \n v: \n {1} \n and \n c: \n {2} \n".format(np.linalg.norm(v-c), v, c))
    
    return c



def attemptLoginToValueInvestingIO():

    # Fill in your details here to be posted to the login form.
    payload = {
        'email': 'nr282@cornell.edu',
        'password': 'ga68bRed4!',
        'api_key': 'c9g3iqe2brtv74e4sth0'
    }

    loginUrl = "https://valueinvesting.io/login"

    # Use 'with' to ensure the session context is closed after use.
    with requests.Session() as s:
        p = s.post(loginUrl, data=payload)
        
        if p.text.find("Wrong password") == -1:
            pass
        else:
            print(p.text.find("Wrong credentials provided..."))
            raise        


def getProjectedFreeCashFlows(ticker):
    
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker))       
    if r.reason != "OK":
        raise RuntimeError("Problem with Request was Found. Error Code {0} \n".format(r.reason))

    
    r = r.json()
    if ticker in set(list(r.keys())) == 0:
        raise RuntimeError("Dictionary does not include ticker {1} is empty: {0} \n".format(ticker, r[ticker]))
    
    if len(r[ticker]) == 0:
        raise RuntimeError("Dictionary for ticker {1} is empty: {0} \n".format(ticker, r[ticker]))

    #It looks like sometimews this could come back empty.
    try:
        projectedFreeCashFlow = r[ticker]["projected_fcf"]    
    except:
        print("Error here: Json dictionary could have problems. {0}".format(r))

    periodsAndFreeCashFlows = []    
    for d in projectedFreeCashFlow:
        fcf = d["fcf"] 
        period = d["period"]
        periodsAndFreeCashFlows.append((period, fcf))
    periodsAndFreeCashFlows.sort(key = lambda x: x[0])
    periodsAndFreeCashFlows = list(map(lambda x: (int(x[0]), int(x[1])), periodsAndFreeCashFlows)) 
    r = requests.get('https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker))
    r = r.json()
    wacc = r[ticker]["wacc_components"]['wacc']
    waccDiscountVector = [(1/(1+wacc)**(i+1)) for i in range(len(periodsAndFreeCashFlows))]

    return periodsAndFreeCashFlows, wacc, waccDiscountVector


def getOutstandingShares(ticker):

    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker))
    r = r.json()
    outstandingShares = r[ticker]["outstanding_share"]
    return outstandingShares


def getNetDebt(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker))
    r = r.json()
    netDebt = r[ticker]["net_debt"]
    return netDebt

def getLongTermGrowthRate(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker))
    r = r.json()
    terminalGrowthRate = r[ticker]["terminal_growth_rate"]
    return terminalGrowthRate


def getDcfFiveYearSharePrice(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker))
    r = r.json()
    sharePrice = r[ticker]["valuation"]["fair_price_dcf_growth_5"]
    return sharePrice

def getSharePrice(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker))
    r = r.json()
    sharePrice = r[ticker]["stock_price"]
    return sharePrice

def getCalculatedDCFAndSharePrice(ticker, numberOfYears = None):        

    periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    longTermGrowthRate = getLongTermGrowthRate(ticker)
    netDebt = getNetDebt(ticker)
    outstandingShares = getOutstandingShares(ticker)  

    if numberOfYears is None:      
        numberOfYears = len(periodsAndFreeCashFlows)
        
    discountedCashFlows = ([(1/((1+wacc)**(i + 1 - 0.5))) * periodsAndFreeCashFlows[i][1] for i in range(numberOfYears)])
    discountedCashFlowValue = sum(discountedCashFlows)

    lastFCFCalculated = (periodsAndFreeCashFlows[numberOfYears-1][1] +
                        periodsAndFreeCashFlows[numberOfYears-2][1] +
                        periodsAndFreeCashFlows[numberOfYears-3][1])/3.0  


    terminalValue = (lastFCFCalculated * (1 + longTermGrowthRate)) / (1+wacc - (1 + longTermGrowthRate))
    terminalValueDiscounted = terminalValue / ((1+ wacc)**numberOfYears)
    totalDcfValue = discountedCashFlowValue + terminalValueDiscounted - netDebt
    sharePrice = totalDcfValue / outstandingShares
    return sharePrice


def getCalculatedValueOfCashFlows(ticker, numberOfYears= None):
    
    periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    longTermGrowthRate = getLongTermGrowthRate(ticker)
    netDebt = getNetDebt(ticker)
    outstandingShares = getOutstandingShares(ticker)  

    if numberOfYears is None:      
        numberOfYears = len(periodsAndFreeCashFlows)
        
    discountedCashFlows = ([(1/((1+wacc)**(i + 1 - 0.5))) * periodsAndFreeCashFlows[i][1] for i in range(numberOfYears)])
    discountedCashFlowValue = sum(discountedCashFlows)

    return discountedCashFlowValue

def getMarketValueOfCashFlows(ticker, numberOfYears = None):
    
    periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    longTermGrowthRate = getLongTermGrowthRate(ticker)
    netDebt = getNetDebt(ticker)
    outstandingShares = getOutstandingShares(ticker)  
    marketSharePrice = getSharePrice(ticker)


    lastFCFCalculated = (periodsAndFreeCashFlows[numberOfYears-1][1] +
                        periodsAndFreeCashFlows[numberOfYears-2][1] +
                        periodsAndFreeCashFlows[numberOfYears-3][1])/3.0  

    terminalValue = (lastFCFCalculated * (1 + longTermGrowthRate)) / (1+wacc - (1 + longTermGrowthRate))
    marketTotalDcfValue = marketSharePrice * outstandingShares
    terminalValueDiscounted = terminalValue / ((1+ wacc)**numberOfYears)
    marketValueOfProjectedCashFlows = marketTotalDcfValue - terminalValueDiscounted + netDebt

    return marketValueOfProjectedCashFlows


def testGetDCFAndSharePrice():

    ticker="XOM"
    expectedXomSharePrice = getDcfFiveYearSharePrice(ticker)    
    calculatedXomSharePrice = getCalculatedDCFAndSharePrice(ticker, numberOfYears = 5)
    print("Calculated Share Price {0} and Expected Share Price {1} \n".format(calculatedXomSharePrice, expectedXomSharePrice))
    assert((np.isclose(np.array(expectedXomSharePrice), np.array(calculatedXomSharePrice), rtol=0.1)).all())
    print("Passed {0} Ticker \n".format(ticker))

    ticker="CVX"
    expectedXomSharePrice = getDcfFiveYearSharePrice(ticker)    
    calculatedXomSharePrice = getCalculatedDCFAndSharePrice(ticker, numberOfYears = 5)
    print("Calculated Share Price {0} and Expected Share Price {1} \n".format(calculatedXomSharePrice, expectedXomSharePrice))
    assert((np.isclose(np.array(expectedXomSharePrice), np.array(calculatedXomSharePrice), rtol=0.1)).all())
    print("Passed {0} Ticker \n".format(ticker))

    ticker="GE"
    expectedGESharePrice = getDcfFiveYearSharePrice(ticker)    
    calculatedGESharePrice = getCalculatedDCFAndSharePrice(ticker, numberOfYears = 5)
    print("Calculated Share Price {0} and Expected Share Price {1} \n".format(calculatedGESharePrice, expectedGESharePrice))
    assert((np.isclose(np.array(expectedGESharePrice), np.array(calculatedGESharePrice), rtol=0.1)).all())
    print("Passed {0} Ticker \n".format(ticker))



def compareValueOfCashFlows(ticker):

    calculatedValueOfCashFlows = getCalculatedValueOfCashFlows(ticker, numberOfYears=5)
    marketValueOfCashFlows = getMarketValueOfCashFlows(ticker, numberOfYears = 5)
    projectedCashFlows, wacc, waccDiscountVector = getProjectedFreeCashFlows(ticker)    

    print("Ticker {2}: \n"
        "Calculated Cash Flow Value \n {0} \n"
        "Expected Cash Flow Value \n {1} \n"
        "Cash Flows are: \n {3} \n"
        "wacc discount vector: {4} \n"
        .format(calculatedValueOfCashFlows, marketValueOfCashFlows, ticker, projectedCashFlows, waccDiscountVector))



def testCompareValueOfCashFlows():

    tickerXOM = "XOM"
    compareValueOfCashFlows(tickerXOM)

    tickerGM = "GM"
    compareValueOfCashFlows(tickerGM)

    tickerPYPL = "PYPL"
    compareValueOfCashFlows(tickerPYPL)

    tickerNVDA = "NVDA"    
    compareValueOfCashFlows(tickerNVDA)

    tickerBP = "BP"
    compareValueOfCashFlows(tickerBP)




def getNASDAQData(ticker, startDate, endDate):
    data = nasdaqdatalink.get("FRED/GDP", start_date=startDate, end_date=endDate)
    return data 


def testValuationAPI(ticker):
    
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker))
    print(r.json())


def testDiscountedCashFlowAPI(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    r = requests.get('https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker))
    print(r.json())



def main():
    testCompareValueOfCashFlows()




main()