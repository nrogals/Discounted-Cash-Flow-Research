import numpy as np
import requests
from scipy.linalg import null_space 
import matplotlib.pyplot as plt


def plotCashFlowVectors():

    wacc = 0.1
    N = 40
    waccVector = np.array([1/(1 + wacc)**(i+1) for i in range(N)])
    waccVector = waccVector[np.newaxis, :]
    k = null_space(waccVector)

    print("Null Space: \n {0} \n".format(k))
    print(k)

    cashFlowVector = np.array([((-1) ** i) * i for i in range(N)])
    cashFlowVector = cashFlowVector[:, np.newaxis]
    alpha = 100
    firstKernelVector = k[:,0]
    secondKernelVector = k[:,1]
    cashFlowVectorPlusFirstKernelVector = cashFlowVector + alpha * firstKernelVector[:, np.newaxis]
    cashFlowVectorPlusSecondKernelVector = cashFlowVector + alpha * secondKernelVector[:, np.newaxis]

   
    valueOfCashFlowVector = np.dot(waccVector, cashFlowVector)
    valueOfCashFlowVectorPlusKernel = np.dot(waccVector, cashFlowVectorPlusFirstKernelVector)

    print("Cash Flow Vector Value {0} \n"
        "Cash Flow Vector Plus First Kernel Vector {1} \n".format(valueOfCashFlowVector, valueOfCashFlowVectorPlusKernel))
    
    v = cashFlowVector[:, 0]
    vPlusFirstKernel = cashFlowVectorPlusFirstKernelVector[:, 0]
    vPlusSecondKernel = cashFlowVectorPlusSecondKernelVector[:, 0]
    print("Value For the First and Second Cash Flows")
    plt.title("Equivalence of Cash-Flows \n Under Linear Discounted Cash-Flow")
    plt.plot(v, label = "Cash Flow Vector")
    plt.plot(vPlusFirstKernel, label = "Cash Flow Vector Plus First Kernel Vector")
    plt.plot(vPlusSecondKernel, label = "Cash Flow Vector Plus Second Kernel Vector")
    plt.legend()
    plt.ylabel('Cash Flow Values')
    plt.xlabel('Period')
    plt.savefig("CashFlowVectorPlusKernel")


plotCashFlowVectors()




