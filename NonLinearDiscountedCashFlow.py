"""
The module here attempts to study the non-linearity found in 
financial market prices. 

The discounted cash-flow can viewed as a linear pricing tool. 

A non-linear discounted cash-flow can be considered, which is what we do here. 

TODO:
DONE    1. Need to investigate the errors that arise from the discounted cash-flow.  
DONE    2. Need to develop a regression to map the the kernel volatility into the error that is seen 
           This will require coming up with a good regression that should be attempt to capture the reliance 
           of the kernel vectors into the potential error. This will represent a non-linear regression. 
DONE    3. A paper was found from Allayannis, Earnings volatility, cash flow volatility, and firm value.
            Darden Graduate School of Business, UVA
                -The paper finds that cash-flow volatility is negatively valued by investors. 
                -Indeed they state that there is approximately 32% decrease in the value of a firm 
                for additional volatility. 
                -Such observations are consistent with risk management theory and suggests managers efforts 
                to produce smooth financial statements may add value to the firm. 
    


    Data Sources For Private Equity 
    1. https://www.pehub.com/us-pe-deal-making-data/


getMarketValueOfCashFlows

Key API: 
https://valueinvesting.io/api/dcf?tickers=GOOGL,AAPL&api_key=

Author: 
Nathaniel Rogalskyj 
"""

import numpy as np
import requests
from scipy.linalg import null_space 
import random   
import logging 
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #for plotting purpose
from sklearn.linear_model import LinearRegression


cashFlowLow = -100
cashFlowHigh = 1000
increment = 100

#I think the limit may be like 20 in a minute. 
#As such, I think that it may be necessary to put a
#10 second delay.
apiDelay = 3

def getResultsFromRequest(requestString):
    #This function deals with the rate limiter that 
    #was put forward by value investing io. 
    time.sleep(apiDelay)
    r = requests.get(requestString)
    return r

def randomWalkIterator():
    current = np.random.uniform(low=cashFlowLow, high=cashFlowHigh)
    while True: 
        current = current + random.randint(-1, 1) * increment
        yield current

def simulateCashFlowsMethod1(numberOfTimePeriods, wacc = 0.1, stdMultiplicativeNoise=0.1):
    """
    Simulates a set of cash flows and penalizes the cash-flows 
    for bad cash-flow volatility based off the the kernel  vector norm.
    """
    
    cashFlowVector = np.array([[next(randomWalkIterator()) for i in range(numberOfTimePeriods)]])
    cashFlowVectorNorm = np.linalg.norm(cashFlowVector)
    
    waccDiscountVector = np.array([[1/(1+wacc)**(i+1) for i in range(numberOfTimePeriods)]])
    #Penalization Vector
    lowerBound = -cashFlowVectorNorm/(numberOfTimePeriods -1)
    upperBound = cashFlowVectorNorm/(numberOfTimePeriods -1)
    penalizationVector = [np.random.uniform(lowerBound,upperBound) for i in range(numberOfTimePeriods - 1)]
    ns = null_space(waccDiscountVector)
    k = np.matmul(ns, penalizationVector)
    kNorm = np.linalg.norm(k)
    modifiedCashFlowVector = cashFlowVector - k
    unmodifiedValue = np.matmul(cashFlowVector, waccDiscountVector.T)
    #Penalize the negative values. 
    #Alpha represents the largest percentage value that can be penalizied. 
    #Cash-Flow Volatility.
    alpha = 0.5
    penalizationValue = -1 * (kNorm / cashFlowVectorNorm) * unmodifiedValue * alpha
    modifiedCashFlowValue = unmodifiedValue - penalizationValue
    stdModifiedCashFlowValue = abs(stdMultiplicativeNoise * modifiedCashFlowValue)
    realNoisyModifiedCashFlowValue = np.random.normal(modifiedCashFlowValue, stdModifiedCashFlowValue)
    return modifiedCashFlowVector, waccDiscountVector, realNoisyModifiedCashFlowValue, modifiedCashFlowValue




def simulateCashFlowsMethod2(numberOfTimePeriods, wacc = 0.1, stdMultiplicativeNoise=0.1):
    """
    TODO: Implement Simulation of Cash Flow Method 2. 
    """
    pass


def simulateCashFlowsMethod3(numberOfTimePeriods, wacc = 0.1, stdMultiplicativeNoise=0.1):
    """
    TODO: Implement Simulation of Cash Flow Method 2. 
    """
    pass




def linearDiscountedCashFlow(w, c):
    w_i = np.array([(1+w)**(-1*(i)) for i in range(len(c))])
    return np.dot(w_i.T, c)



def nonLinearDiscountedCashFlow(w,c):
    w_i = np.array([[(1+w)**(-1*(i)) for i in range(len(c))]])   
    #print("w_i {0} \n".format(w_i))

    ns = null_space(w_i)    
    q,r = np.linalg.qr(w_i)
    angle = np.dot(ns[:,1].T, ns[:,0])
    
    print("Null Space is provided by: \n {0} for the linear map {1}. \n It has the dimensions of {2} \n".format(ns, w_i, ns.shape))
    
    assert(np.isclose(angle, 0.0))
    print("Matrix QR {0} and WI {1} \n".format(q*r, w_i))    
    assert(np.isclose(np.linalg.norm(q*r - w_i), 0.0))
    
    projWiAgainstNs = np.matmul(w_i, ns)
    projCAgainstNs = np.matmul(c, ns)        
    wINorm = np.linalg.norm(w_i)
    projCAgainstWi = np.matmul(c, w_i.T / wINorm)      
    #k = SUM_i=1^N-1 <c, ns[:,i]> * ns[:,i]
    k = np.matmul(ns, projCAgainstNs[:, np.newaxis])
    r = np.matmul(w_i.T / np.linalg.norm(w_i.T), projCAgainstWi[:,np.newaxis])
    v = k + r
    v = v[:,0]

    print("Kernel Vector \n {0} \n Range Vector \n {1} \n Full Vector \n {2}".format(k, r, k + r))

    print("projCAgainstNs has the shape {0}".format(projCAgainstNs.shape))

    return k, r, v, ns, projCAgainstNs, w_i


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
        

def parseToDate(date):
    #What should I aim to do with the date? 
    #I will round up the dates. 

    from datetime import datetime
    n = datetime.now()
    currentYear = n.year

   

    if("M" in date):
        providedYear = int(date.split("/")[-1])
        if providedYear <= currentYear: 
            print("Returning None Since Current Year {0} is greater or equal to Provided Year {1} \n".format(providedYear, currentYear))
            return None
        else:
            return providedYear

    else:
        try:
            providedYear = int(date)
            return providedYear
        except:
            pass
    

def getProjectedFreeCashFlows(ticker):
    
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)

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
        raise Exception("Error here: Json dictionary could have problems. {0}".format(r))

    periodsAndFreeCashFlows = []    
    for d in projectedFreeCashFlow:
        fcf = d["fcf"] 
        period = d["period"]
        periodsAndFreeCashFlows.append((period, fcf))
    periodsAndFreeCashFlows.sort(key = lambda x: x[0])
    periodsAndFreeCashFlows = list(map(lambda x: (parseToDate(x[0]), int(x[1])) if parseToDate(x[0]) is not None else None, periodsAndFreeCashFlows)) 
    periodsAndFreeCashFlows = list(filter(lambda x: x is not None, periodsAndFreeCashFlows))

    requestString = 'https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)
    r = r.json()
    wacc = r[ticker]["wacc_components"]['wacc']
    waccDiscountVector = [(1/(1+wacc)**(i+1)) for i in range(len(periodsAndFreeCashFlows))]

    return periodsAndFreeCashFlows, wacc, waccDiscountVector


def getOutstandingShares(ticker):

    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)
    r = r.json()
    outstandingShares = r[ticker]["outstanding_share"]
    return outstandingShares


def getNetDebt(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)
    r = r.json()
    netDebt = r[ticker]["net_debt"]
    return netDebt

def getLongTermGrowthRate(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)
    r = r.json()
    terminalGrowthRate = r[ticker]["terminal_growth_rate"]
    return terminalGrowthRate

def getDcfFiveYearSharePrice(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)
    r = r.json()
    sharePrice = r[ticker]["valuation"]["fair_price_dcf_growth_5"]
    return sharePrice

def getSharePrice(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)
    r = r.json()
    sharePrice = r[ticker]["stock_price"]
    return sharePrice

def getCalculatedDCFAndSharePrice(ticker, numberOfYears = None):        

    try:
        periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    except:
        logging.INFO("Projected Free Cash Flows Not Available")
        raise RuntimeError("Projected Free Cash Flows are not available. There is an incorrect ticker.")
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


def getRelevantShareCharacteristics(ticker):

    periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    longTermGrowthRate = getLongTermGrowthRate(ticker)
    netDebt = getNetDebt(ticker)
    outstandingShares = getOutstandingShares(ticker)  
    marketSharePrice = getSharePrice(ticker)

    return periodsAndFreeCashFlows, longTermGrowthRate, netDebt, outstandingShares, marketSharePrice


def getMarketValueOfCashFlows(ticker, marketSharePrice = None, numberOfYears = None):
    """
    The Market Value of the CashFlows can be looked at as the value of the 
    Projection Period provided by:
    https://valueinvesting.io/XOM/valuation/dcf-growth-exit-5y
    """
    
    periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    longTermGrowthRate = getLongTermGrowthRate(ticker)
    netDebt = getNetDebt(ticker)
    outstandingShares = getOutstandingShares(ticker)
    
    if marketSharePrice is None: 

        marketSharePrice = getSharePrice(ticker)
    

    if numberOfYears is None:
        numberOfYears = len(periodsAndFreeCashFlows)


    lastFCFCalculated = (periodsAndFreeCashFlows[numberOfYears-1][1] +
                        periodsAndFreeCashFlows[numberOfYears-2][1] +
                        periodsAndFreeCashFlows[numberOfYears-3][1])/3.0  

    terminalValue = (lastFCFCalculated * (1 + longTermGrowthRate)) / (1+wacc - (1 + longTermGrowthRate))
    marketTotalDcfValue = marketSharePrice * outstandingShares
    terminalValueDiscounted = terminalValue / ((1+ wacc)**numberOfYears)
    #I think that this is very important to the calculation.
    #Where does netdebt play in. 
    marketValueOfProjectesCashFlows = marketTotalDcfValue - terminalValueDiscounted + netDebt

    return marketValueOfProjectesCashFlows


def testGetDCFAndSharePrice():

    #Reduce this tolerance. I think that this should be closer to zero.
    allowableTolerance = 0.3
    
    ticker="XOM"
    expectedXomSharePrice = getDcfFiveYearSharePrice(ticker)    
    calculatedXomSharePrice = getCalculatedDCFAndSharePrice(ticker, numberOfYears = 5)
    print("Calculated Share Price {0} and Expected Share Price {1} \n".format(calculatedXomSharePrice, expectedXomSharePrice))
    assert((np.isclose(np.array(expectedXomSharePrice), np.array(calculatedXomSharePrice), rtol=allowableTolerance)).all())
    print("Passed {0} Ticker \n".format(ticker))
    
    ticker="CVX"
    expectedCvxSharePrice = getDcfFiveYearSharePrice(ticker)    
    calculatedCvxSharePrice = getCalculatedDCFAndSharePrice(ticker, numberOfYears = 5)
    print("Calculated Share Price {0} and Expected Share Price {1} \n".format(calculatedCvxSharePrice, expectedCvxSharePrice))
    assert((np.isclose(np.array(expectedCvxSharePrice), np.array(calculatedCvxSharePrice), rtol=allowableTolerance)).all())
    print("Passed {0} Ticker \n".format(ticker))
    
    
    ticker="GE"
    expectedGESharePrice = getDcfFiveYearSharePrice(ticker)    
    calculatedGESharePrice = getCalculatedDCFAndSharePrice(ticker, numberOfYears = 5)
    print("Calculated Share Price {0} and Expected Share Price {1} \n".format(calculatedGESharePrice, expectedGESharePrice))
    assert((np.isclose(np.array(expectedGESharePrice), np.array(calculatedGESharePrice), rtol=allowableTolerance)).all())
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

def testNonLinearDiscountedCashFlowOnMarketData(ticker):

    calculatedValueOfCashFlows = getCalculatedValueOfCashFlows(ticker, numberOfYears=5)
    marketValueOfCashFlows = getMarketValueOfCashFlows(ticker, numberOfYears = 5)
    projectedCashFlows, wacc, waccDiscountVector = getProjectedFreeCashFlows(ticker) 


    d = [p[0] for p in projectedCashFlows]
    c = [p[1] for p in projectedCashFlows]
    print("Dates: \n {0} \n".format(d))
    print("Cash Flows \n {0} \n".format(c))

    k, r, v, ns, projCAgainstNs, _ = nonLinearDiscountedCashFlow(wacc,c) 
    print("K Vector {0}, R vector {1} V {2}, NS {3}, ProjCAgainstNS {3}")

    print("Null Space {0} with shape {1} \n and Projection of C onto Null Space {2} \n".format(ns, ns.shape, projCAgainstNs))

    #TODO:
    #Understand why the plotting of the null-space vectors 
    #does not appropiately plot the correct values. 
    n = len(ns)
    import matplotlib.pyplot as plt
    plt.plot(d, ns, 'o')
    plt.ylabel('Cash-Flow Amount')   
    plt.xlabel("Date") 
    plt.title("Plot of Null Space of Discount Vector For Ticker {0}".format(ticker))
    plt.legend()
    plt.savefig("NullSpaceOfDiscountVectorForTicker{0}.png".format(ticker))
    plt.clf()
    #We can also plot the coordinate vectors here? 
    #There are N-1 coordinate vectors. 
    #Representing the different elements of the kernel.
    print("Size of Projection of C Against Null Space NS {0} \n".format(projCAgainstNs))
    plt.plot([i for i in range(len(projCAgainstNs))], list(projCAgainstNs),  marker='o')
    plt.title('Different Projections Of K onto NS For Ticker {0}'.format(ticker))
    plt.legend()
    plt.savefig("PlotOfProjectionsOfCashFlowOntoNullSpaceForTicker{0}.png".format(ticker))

    return calculatedValueOfCashFlows, marketValueOfCashFlows



def getFeatureVector(cashFlowVector, wacc):
    
    k, r, v, ns, projCAgainstNs, waccVector = nonLinearDiscountedCashFlow(wacc, cashFlowVector)

    cashFlowVector = cashFlowVector[:, np.newaxis]
    linearDiscountedValue = np.matmul(waccVector, cashFlowVector)

    #These values can be used in the calculation. 
    #projCAgainstNs are the projection of C against the null-space. 
    absoluteValueOfProjections = abs(projCAgainstNs)[np.newaxis, :]


    #Need to extract the relevant attributes from the nonLinearDiscountedCashFlow.
    fullFeatureVector = absoluteValueOfProjections
    
    return fullFeatureVector, linearDiscountedValue


def createFeatureMatrix(cashFlowMatrix, waccVector):


    if len(waccVector.shape) == 2:
        waccVector = waccVector.flatten()
    
    #Need to create feature matrix
    n,p = cashFlowMatrix.shape
    featureMatrix = []
    linearDiscountedCashFlowValues = []
    for i in range(n):
        wacc = waccVector[i]
        cashFlowVector = np.array(cashFlowMatrix[i, :]) #Need to cast into two d array. 1 x N array.
        #For some reason, these values look off.
        featureVector, linearDiscountedCashFlowValue = getFeatureVector(cashFlowVector, wacc)
        featureMatrix.append(featureVector.flatten())
        linearDiscountedCashFlowValues.append(linearDiscountedCashFlowValue.flatten())
    featureMatrix = np.array(featureMatrix)
    linearDiscountedCashFlowValues = np.array(linearDiscountedCashFlowValues)
    return featureMatrix, linearDiscountedCashFlowValues


def simulateData():

    numberOfTimePeriods = 10 
    lowWacc = 0.0
    highWacc = 0.3
    stdMultiplicativeNoise = 0.1


    #Append or concat the matrix.
    cashFlowMatrix = []
    cashFlowPrice = []
    waccVector = []
    numberOfIterates = 1000
    for i in range(numberOfIterates):
        wacc = random.uniform(lowWacc, highWacc)
        modifiedCashFlowVector, waccDiscountVector, realNoisyModifiedCashFlowValue, modifiedCashFlowValue = simulateCashFlowsMethod1(numberOfTimePeriods, wacc = wacc, stdMultiplicativeNoise=stdMultiplicativeNoise)
        cashFlowMatrix.append(modifiedCashFlowVector)
        cashFlowPrice.append(realNoisyModifiedCashFlowValue)
        waccVector.append(wacc)

    waccVector = np.array(waccVector)
    cashFlowMatrix = np.concatenate(cashFlowMatrix, axis = 0)
    cashFlowPrice = np.concatenate(cashFlowPrice, axis = 0)

    return waccVector, cashFlowMatrix, cashFlowPrice


def getRealData():

    import pandas as pd
    logging.info("Getting Real Data \n")
    apiKey = "5KaYTqFoTFjUIjtv1SUUxP_2TTaAJp2j" 

    import os
    cwd = os.getcwd()
    newDirectory = "realDataStored"
    newPath = os.path.join(cwd, newDirectory)

    if(not os.path.exists(newPath)):
        os.mkdir(newPath)
    
    cashFlowMatrixDfPath = os.path.join(newPath, "cashFlowMatrixDf.csv")
    cashFlowPriceDfPath = os.path.join(newPath, "cashFlowPriceDf.csv")
    waccVectorDfPath = os.path.join(newPath, "waccVectorDf.csv")
    
    realDataPathsExist = (os.path.exists(cashFlowMatrixDfPath) & os.path.exists(cashFlowPriceDfPath) & os.path.exists(waccVectorDfPath))

    if realDataPathsExist:
        logging.info("Getting Real Data From Disk \n")
        cashFlowMatrixDf = pd.read_csv(cashFlowMatrixDfPath)
        cashFlowPriceDf = pd.read_csv(cashFlowPriceDfPath)
        waccVectorDf = pd.read_csv(waccVectorDfPath)
        cashFlowMatrix = cashFlowMatrixDf.to_numpy()
        cashFlowPrice = cashFlowPriceDf.to_numpy()
        waccVector = waccVectorDf.to_numpy()
        logging.info("Pulled Real Data From Disk \n")

        return waccVector, cashFlowMatrix, cashFlowPrice

    logging.info("Real Data Does Not Exist On Disk \n")
    #Get tickers here.
    r = None
    try:
        logging.info("Tickers are being loaded...\n")
        #/v2/snapshot/locale/us/markets/stocks/tickers
        requestString = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey={0}".format(apiKey)
        logging.info("Request: {0} \n".format(requestString))
        r = requests.get(requestString)
    except Exception as e:
        print(e)

    print("Request: \n")
    print(r)

    import json
    r = json.loads(r.content.decode('utf-8'))
    tickers = [record["ticker"] for record in r["tickers"] if not ("." in record["ticker"])]
    
    cashFlowMatrix = []
    cashFlowPriceVector = []
    waccVector = []
    tickerList = []

    try: 
        for ticker in tickers: 
            try: 
                logging.info("Get Free Cash-Flow Projections For Ticker {0} \n".format(ticker))
                periodsAndFreeCashFlows, wacc, waccDiscountVector = getProjectedFreeCashFlows(ticker)
            except Exception as e: 
               logging.error(e)
               logging.exception(e)
               logging.info("Cash-Flows Not Available For the ticker {0} \n".format(ticker))
               continue

            try:
                logging.info("Get Market Value of Cash Flows for the ticker {0} \n".format(ticker))
                marketValueOfCashFlowValue = getMarketValueOfCashFlows(ticker)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logging.info("Exception is provided by: {0} \n".format(e))
                logging.info("Cash-Flows Not Available For the ticker {0} \n".format(ticker))
                continue

            cashFlows = [x[1] for x in periodsAndFreeCashFlows]
            logging.info("**************************** \n"
                        "Resolve: Calculated CashFlow Matrix \n"
                        "CashFlow Price \n"
                        "WACC Vector \n"
                        "***************************** \n")
            cashFlowMatrix.append(cashFlows)
            cashFlowPriceVector.append(marketValueOfCashFlowValue)
            waccVector.append(wacc)
            tickerList.append(ticker)
    except: 
        logging.error("Error was encountered in the try-block \n")
    finally:
        #Filter for the array that has the largest number of arrays 
        #of arrays
        lengthOfCashFlowVectorList = [len(cashFlowVector) for cashFlowVector in cashFlowMatrix]
        from collections import Counter 
        c = Counter(lengthOfCashFlowVectorList)
        mostCommonLength = c.most_common(1)[0][0]
        assert(type(mostCommonLength) == int)

        cashFlowMatrixFiltered = []
        cashFlowPriceFiltered = []
        waccVectorFiltered = []
        tickerListFiltered = []
        for i in range(len(cashFlowMatrix)):

            cashFlow = cashFlowMatrix[i]
            cashFlowPrice = cashFlowPriceVector[i]
            wacc = waccVector[i]
            ticker = tickerList[i]

            if len(cashFlow) == mostCommonLength:
                cashFlowMatrixFiltered.append(cashFlow)
                cashFlowPriceFiltered.append(cashFlowPrice)
                waccVectorFiltered.append(wacc)
                tickerListFiltered.append(ticker)

    
        #Convert the real data matricies to
        #numpy matricies
        logging.info("Converting to Numpy Arrays \n")
        cashFlowMatrix = np.array(cashFlowMatrixFiltered)
        cashFlowPrice = np.array(cashFlowPriceFiltered)
        waccVector = np.array(waccVectorFiltered)

        #Store the cashFlowMatrix, cashFlowPrice, and waccVector to
        #be used in the future. 

        #Need to figure out how to store these numpy matricies 
    
        cashFlowMatrixDf = pd.DataFrame(cashFlowMatrix)
        cashFlowPriceDf = pd.DataFrame(cashFlowPrice)
        waccVectorDf = pd.DataFrame(waccVector)

        logging.info("Writing Real Data To Disk \n")
        cashFlowMatrixDf.to_csv(cashFlowMatrixDfPath, index=False)
        cashFlowPriceDf.to_csv(cashFlowPriceDfPath, index=False)
        waccVectorDf.to_csv(waccVectorDfPath, index=False)


        return waccVector, cashFlowMatrix, cashFlowPrice


def testNonLinearDiscountedCashFlowOnRealData(plotResiduals = True):
    
    waccVector, cashFlowMatrix, cashFlowPrice = getRealData()


    featureMatrix, linearDiscountedCashFlowValues = createFeatureMatrix(cashFlowMatrix, waccVector)
    errorInCashFlowPrice = cashFlowPrice - linearDiscountedCashFlowValues
    try:
        #We want a linear least squares solve.
        #np.linalg.lstsq. 
        logging.info("Running np.linalg.lstsq on real data \n")
        import numpy as np
        import pandas as pd
        n, p = cashFlowMatrix.shape
        featureMatrixDf = pd.DataFrame(cashFlowMatrix, columns = ["Projection_Onto_Kernel_Vector_" + str(i) for i in range(p)])
        errorInCashFlowDf = pd.DataFrame(errorInCashFlowPrice, columns = ["ErrorInCashFlow"])
        regressionDf = pd.concat([featureMatrixDf, errorInCashFlowDf], axis=0)
        regressionDf.to_csv("RegressionDf.csv")

        x, residuals, rank, s = np.linalg.lstsq(featureMatrix,errorInCashFlowPrice)
        logging.info("After running np.linalg.lstsq on real data \n")

        with open("linearLeastSquaresNonLinearDiscounteddCashFlow.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(featureMatrix))
            print("Error In Cash Flow Values \n {0} \n".format(errorInCashFlowPrice))
            print("Vector X is provided by: \n {0} \n".format(x), file=external_file)
            print("Residuals R is provided by: \n {0} \n".format(residuals), file=external_file)
            print("Rank is provided by: \n {0} \n".format(rank), file=external_file)
            print("Singular Values is provided by: \n {0} \n".format(s), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)



    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt  #for plotting purpose
        from sklearn.linear_model import LinearRegression

        X = featureMatrix
        Y = errorInCashFlowPrice

        reg = LinearRegression().fit(X, Y)
        coeff = reg.coef_
        intercept = reg.intercept_ 

        if plotResiduals:
            residualsVector = Y - np.matmul(X, coeff.T)
            residualsVector = residualsVector.flatten()
            residualsIndex = [i for i in range(len(residualsVector))]
            plt.hist(residualsVector)
            plt.show()
        
        with open("standardNonLinearDiscountedCashFlow.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(featureMatrix))
            print("Error In Cash Flow Values \n {0} \n".format(errorInCashFlowPrice))
            print("Vector X is provided by: \n {0} \n".format(coeff), file=external_file)
            print("Intercept is provided by: \n {0} \n".format(intercept), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)
   

    try: 
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt  #for plotting purpose
        import scipy
        import os

        #Loop over the different kernel vectors.
        logging.info("\n Calculating Projections Onto Kernel Vectors \n")
        n, p = X.shape
        for i in range(p):
            x = X[:,i]
            y = Y[:,0]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            logging.info("\n Projection Against {0}th Kernel Vector \n".format(i))
            plt.scatter(x,y)
            plt.ylabel("Error In Cash-Flows")
            plt.xlabel("Projection Against {0}th Kernel Vector".format(i))
            plt.title("Error in Price Plotted Against Cash-Flow Volatility")
            directory_name = "kernel_vector_projections"
            if not os.path.exists(directory_name): #What does the not operator look like? 
                logging.info("Directory {0} does not exist, so make it \n".format(directory_name))
                os.mkdir(directory_name)
            else:
                pass
            
            
            path_to_file = os.path.join(directory_name, "plot{0}thKernelVectorAgainstErrorInCashFlows.png".format(i))
            if os.path.exists(path_to_file):
                logging.info("Since the figure exists, it is necessary to remove it \n The path being removed is: \n {0}".format(path_to_file))
                os.remove(path_to_file)
            else:
                logging.info("Saving figure to path \n {0} \n".format(path_to_file))
                plt.savefig(path_to_file)

            logging.info("Closing plots since they are no longer needed \n")
            #We are now done plotting. So we can close
            #the plots and clear them. 
            plt.clf()
            plt.close()

        with open("standardNonLinearDiscountedCashFlowWithOneVariable.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(x))
            print("Error In Cash Flow Values \n {0} \n".format(y))
            print("Slope is provided by: \n {0} \n".format(slope), file=external_file)
            print("Intercept is provided by: \n {0} \n".format(intercept), file=external_file)
            print("R Value is provided by: \n {0} \n".format(r_value), file=external_file)
            print("Std Err is provided by: \n {0} \n".format(std_err), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Plotting \n")
        logging.error(e)

    

    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt  #for plotting purpose
        from sklearn.linear_model import LinearRegression

        X =featureMatrix[:,0:3]
        Y = errorInCashFlowPrice

        reg = LinearRegression().fit(X, Y)
        coeff = reg.coef_
        intercept = reg.intercept_      

        #More like subset of kernel vectors
        with open("standardNonLinearDiscountedCashFlowFirstKernelVector.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(X))
            print("Error In Cash Flow Values \n {0} \n".format(errorInCashFlowPrice))
            print("Vector X is provided by: \n {0} \n".format(coeff), file=external_file)
            print("Intercept is provided by: \n {0} \n".format(intercept), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)


    try:
        logging.info("\n Running Lasso Regression on Real Data \n")
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt  #for plotting purpose
        from sklearn.linear_model import Lasso

        X = featureMatrix
        Y = errorInCashFlowPrice

        reg = Lasso(alpha=0.1).fit(X, Y)
        coeff = reg.coef_
        intercept = reg.intercept_
        
        if plotResiduals:
            residualsVector = Y - np.matmul(X, coeff.T)
            residualsVector = residualsVector.flatten()
            residualsIndex = [i for i in range(len(residualsVector))]
            plt.hist(residualsVector)
            plt.savefig('residualsHistogramPlotForLasso.png')

        with open("lassoNonLinearRealData.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(X))
            print("Error In Cash Flow Values \n {0} \n".format(errorInCashFlowPrice))
            print("Vector X is provided by: \n {0} \n".format(coeff), file=external_file)
            print("Intercept is provided by: \n {0} \n".format(intercept), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)



def testNonLinearDiscountedCashFlowOnSimulatedData():

    waccVector, cashFlowMatrix, cashFlowPrice = simulateData()
    featureMatrix, linearDiscountedCashFlowValues = createFeatureMatrix(cashFlowMatrix, waccVector)
    errorInCashFlowPrice = cashFlowPrice - linearDiscountedCashFlowValues
    try:
        #We want a linear least squares solve.
        #np.linalg.lstsq. 
        x, residuals, rank, s = np.linalg.lstsq(featureMatrix,errorInCashFlowPrice)

        with open("llsqSimulatedDataNonLinearDiscounteddCashFlow.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(featureMatrix))
            print("Error In Cash Flow Values \n {0} \n".format(errorInCashFlowPrice))
            print("Vector X is provided by: \n {0} \n".format(x), file=external_file)
            print("Residuals R is provided by: \n {0} \n".format(residuals), file=external_file)
            print("Rank is provided by: \n {0} \n".format(rank), file=external_file)
            print("Singular Values is provided by: \n {0} \n".format(s), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)



    try:
        
        X = featureMatrix
        Y = errorInCashFlowPrice

        reg = LinearRegression().fit(X, Y)
        coeff = reg.coef_
        intercept = reg.intercept_      

        with open("lrOnSimulatedDataNonLinearDiscountedCashFlow.txt", "w") as external_file:
            print("Feature Matrix \n {0} \n".format(featureMatrix))
            print("Error In Cash Flow Values \n {0} \n".format(errorInCashFlowPrice))
            print("Vector X is provided by: \n {0} \n".format(coeff), file=external_file)
            print("Intercept is provided by: \n {0} \n".format(intercept), file=external_file)
            external_file.close()

    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)


    


def testMarketValueOfCashFlows():

    print("Market Value of Cash Flows \n")
    ticker = "XOM"
    marketSharePrice = 128 
    calculatedMarketValueOfCashFlows = getMarketValueOfCashFlows(ticker, marketSharePrice = marketSharePrice)
    #My calculation of the market value of the cash flows
    #is provided below.

    #In the below lines we look at calculating the appropiate value for the 
    #market value of xom's cash flows. 
    periodsAndFreeCashFlows, longTermGrowthRate, netDebt, outstandingShares, marketSharePrice = getRelevantShareCharacteristics(ticker)    

    print("Market Value Of Cash Flows {0} \n".format(calculatedMarketValueOfCashFlows))
    
    
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


def testValuationAPI(ticker):
    
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/valuation?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)


def testDiscountedCashFlowAPI(ticker):
    apiKey = "c9g3iqe2brtv74e4sth0"
    requestString = 'https://valueinvesting.io/api/dcf?tickers={1}&api_key={0}'.format(apiKey, ticker)
    r = getResultsFromRequest(requestString)

def main():

    
   logging.info("Testing on Real Data \n")
   testNonLinearDiscountedCashFlowOnRealData()
    

main()