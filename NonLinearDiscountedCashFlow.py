"""
The module here attempts to study the non-linearity found in 
financial market prices. 

The discounted cash-flow can viewed as a linear pricing tool. 

A non-linear discounted cash-flow can be considered, which is what we do here. 

TODO:
    1. Need to investigate the errors that arise from the discounted cash-flow.  
    2. Understand why the null space vector plotting does not accurately represent 
    the values that get assigned in the charts. It looks like the plotting is wrong.
    3. Need to develop a simiulator for cash-flow volatility that we can then invert for and 
    see how our algorithms do on simulated data. 
    4. Need to develop a regression to map the the kernel volatility into the error that is seen 
    This will require coming up with a good regression that should be attempt to capture the reliance 
    of the kernel vectors into the potential error. This will represent a non-linear regression. 
    5. Analyze other research that provide one with a view on cash-flow volatility and its impact on 
    the pricing of the asset with a fixed wacc. 
    6. I may want to change the style of the code so that all vectors are column or row vectors so that 
    it is not confusing to the user. 

    
    1. Paper Finds the Following:
        - Empirical Evidence that cash flow volatility is negatively valued by investors. 
        - The magnitude of the effect is substantial with a one standard deviation increase in cash-flow 
        volatility resulting in approximately 32% decrease in the value of the firm. This is consistent with 
        risk management theory and suggests that managers efforts to produce smooth financial statements may 
        add value to the firm.


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

def getMarketValueOfCashFlows(ticker, numberOfYears = None):
    
    periodsAndFreeCashFlows, wacc, _ = getProjectedFreeCashFlows(ticker)     
    longTermGrowthRate = getLongTermGrowthRate(ticker)
    netDebt = getNetDebt(ticker)
    outstandingShares = getOutstandingShares(ticker)  
    marketSharePrice = getSharePrice(ticker)

    if numberOfYears is None:
        numberOfYears = len(periodsAndFreeCashFlows)


    lastFCFCalculated = (periodsAndFreeCashFlows[numberOfYears-1][1] +
                        periodsAndFreeCashFlows[numberOfYears-2][1] +
                        periodsAndFreeCashFlows[numberOfYears-3][1])/3.0  

    terminalValue = (lastFCFCalculated * (1 + longTermGrowthRate)) / (1+wacc - (1 + longTermGrowthRate))
    marketTotalDcfValue = marketSharePrice * outstandingShares
    terminalValueDiscounted = terminalValue / ((1+ wacc)**numberOfYears)
    marketValueOfProjectedCashFlows = marketTotalDcfValue - terminalValueDiscounted + netDebt

    return marketValueOfProjectedCashFlows


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

    k, r, v, ns, projCAgainstNs = nonLinearDiscountedCashFlow(wacc,c) 
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
    fullFeatureVector = np.concatenate([linearDiscountedValue, absoluteValueOfProjections], axis = 1)
    
    return fullFeatureVector


def createFeatureMatrix(cashFlowMatrix, waccVector):
    
    #Need to create feature matrix
    n,p = cashFlowMatrix.shape
    featureMatrix = []
    for i in range(n):
        wacc = waccVector[i]
        cashFlowVector = np.array(cashFlowMatrix[i, :]) #Need to cast into two d array. 1 x N array.
        featureVector = getFeatureVector(cashFlowVector, wacc)
        featureMatrix.append(featureVector)

    featureMatrix = np.concatenate(featureMatrix, axis = 0)
    return featureMatrix



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
    
    apiKey = "5KaYTqFoTFjUIjtv1SUUxP_2TTaAJp2j"
    
    #Get tickers here.
    r = None
    try:
        r = requests.get("https://api.polygon.io/v3/reference/tickers?apiKey={0}".format(apiKey))
    except Exception as e:
        print(e)

    print("Request: \n")
    print(r)

    import json
    r = json.loads(r.content.decode('utf-8'))
    tickers = [record["ticker"] for record in r["results"] if (record["market"] == "stocks") and not ("." in record["market"])]
    
    cashFlowMatrix = []
    cashFlowPrice = []
    waccVector = []

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
        cashFlowPrice.append(marketValueOfCashFlowValue)
        waccVector.append(wacc)

    return waccVector, cashFlowMatrix, cashFlowPrice


def testNonLinearDiscountedCashFlowOnRealData():
    
    waccVector, cashFlowMatrix, cashFlowPrice = getRealData()
    featureMatrix = createFeatureMatrix(cashFlowMatrix, waccVector)

    try:
        #We want a linear least squares solve.
        #np.linalg.lstsq. 
        x, residuals, rank, s = np.linalg.lstsq(featureMatrix,cashFlowPrice)
    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)
    
    print("Vector X is provided by: {0} \n".format(x))
    print("Residuals R is provided by: {0} \n".format(residuals))
    print("Rank is provided by: {0} \n".format(rank))
    print("Singular Values is provided by: {0} \n".format(s))
    



def  testNonLinearDiscountedCashFlowOnRealData():

    
    waccVector, cashFlowMatrix, cashFlowPrice = getRealData()
    featureMatrix = createFeatureMatrix(cashFlowMatrix, waccVector)


    try:
        #We want a linear least squares solve.
        #np.linalg.lstsq. 
        x, residuals, rank, s = np.linalg.lstsq(featureMatrix,cashFlowPrice)
    except Exception as e:
        logging.error("Error in Least Squares Solve: \n")
        logging.error(e)
    
    print("Vector X is provided by: {0} \n".format(x))
    print("Residuals R is provided by: {0} \n".format(residuals))
    print("Rank is provided by: {0} \n".format(rank))
    print("Singular Values is provided by: {0} \n".format(s))

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
    
    #ticker = "XOM"
    #calculatedValueOfCashFlows, marketValueOfCashFlows = calculateMarketValues(ticker)
    #testGetDCFAndSharePrice()

    #testNonLinearDiscountedCashFlowOnSimulatedData()

    #numberOfTimePeriods = 10
    #cashFlowVector, waccDiscountVector, realModifiedCashFlowValue, modifiedCashFlowValue = simulateCashFlowsMethod1(numberOfTimePeriods)


    #print("Cash Flow Vector {0}, wacc Discount Vector {1}, realModifiedCashFlowValue {2}, modifiedCashFlowValue {3} \n".format(cashFlowVector,
    #                                                                                                                          waccDiscountVector, 
    #                                                                                                                          realModifiedCashFlowValue, 
    #                                                                                                                          modifiedCashFlowValue))
    



    getRealData()
    r = 2 + 2



    



main()