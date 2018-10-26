import numpy as np
import pandas as pd
import glob
import time
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tempfile import TemporaryFile
from PIL import Image

lastDaysAmount = 90

def toCorrCoef(x,y,lastDaysAmount,xlastBoundary,ylastBoundary,coefType):
    coef = 0
    if coefType == "Standart":
        coef = np.corrcoef(x[xlastBoundary - lastDaysAmount:xlastBoundary], y[ylastBoundary - lastDaysAmount:ylastBoundary])
        coef = coef[1,0]
    elif coefType == "Spearmanm":
        coef = spearmanr(x[xlastBoundary - lastDaysAmount:xlastBoundary],
                         y[ylastBoundary - lastDaysAmount:ylastBoundary])[0]
    mean = np.mean(x[xlastBoundary - lastDaysAmount:xlastBoundary])  # to calculate mean of X array
    return coef,mean

def centralityOfPoint(v,adjmatrix):
    sum = 0
    n = len(adjmatrix[0])
    for u in range(n):
        if u != v :
            sum = sum + adjmatrix[u][v]
    sum = sum / (n - 1)
    return sum
def __abs__centralityOfPoint(v,adjmatrix):
    sum = 0
    n = len(adjmatrix[0])
    for u in range(n):
        if u != v:
            sum = sum + abs(adjmatrix[u][v])
    sum = sum / (n - 1)
    return sum
def __calcCentralityVect(adjmatrix,withAbs):
    size = len(adjmatrix[0])
    vect =  np.zeros(size)
    for v in range(size):
        if withAbs == False:
            vect[v] = centralityOfPoint(v,adjmatrix)
        else:
            vect[v] = __abs__centralityOfPoint(v,adjmatrix)
    return vect
def readAllFile(filepath):
    fileName,fileValue = [],[]
    for filename in glob.glob(filepath):
        spy = pd.read_csv(filename)
        tempFile = spy['Adj Close'].values.astype(float)
        fileValue.append(tempFile)
        tempName = filename.replace(".csv", "").split("\\")
        fileName.append(tempName[1])
    return fileName,fileValue

def correToGraphLength(cx,mean):
    if cx >= 0:
        return 1/(cx)
    else :
        return 1/(cx)

"""
   the first parameter of timeSeriesList is given by input to indicate time that outside want to receive.
   the second parameter of timeSeriesList is given by input to show shifting corelation matrix
   If maxshiftingAmount is zero,the list won't return to second dimension array
"""
def calcAdjCorrMatrix(x,maxshifting,maxshiftingAmount,coefType):
    timeSeriesList = []

    for t in range(0,maxshifting,1):
        timeSeriesList.append([])
        for shiftingAmount in range(0,maxshiftingAmount,1):
            timeSeriesList[t].append([])
            correlationMatrix = np.zeros((len(fileValue), len(fileValue)))
            for i in range(len(x)):
                for j in range(len(x)):
                    lastXBoundary = len(x[i])-t - shiftingAmount
                    lastYBoundary = len(x[j])-t
                    corrEf,mean = toCorrCoef(x[i],x[j], lastDaysAmount,lastXBoundary,lastYBoundary,coefType)
                    correlationMatrix[i][j] = corrEf
            timeSeriesList[t][shiftingAmount].append(correlationMatrix)
    return timeSeriesList
"""""
This method receive a input that identified as a shited correlation
And it return a matrix that its dimension is reduced  while calculating its periodic max values
For example: 
            (100,5,3,3) 
            The first index is that calculated time series amount of the data by passing the last days
            The second index is indicate that how we could get amount from the last index of now day
            The third index and last index show that correlation matrix of financial datas.Hence,its dimension 
                directly related to how many types of stock prices is received from dataset
"""""
def maxCorrPeriodic(matrix):
    result = []
    for i in range(matrix.shape[0]):
        temp = np.zeros((matrix.shape[2], matrix.shape[3]))
        tempMax = np.zeros((matrix.shape[2], matrix.shape[3]))
        tempMin = np.zeros((matrix.shape[2],matrix.shape[3]))
        for j in range(matrix.shape[1]):
            tempMax = np.maximum(tempMax,matrix[i][j])
            tempMin = np.minimum(tempMin,matrix[i][j])
        """"
        This block calculate which of the value bigger than other
        By the time code is coming this block,calculated a max and min corelation matrix.After that, this code return once unique matrix 
        """
        for i in range(0,matrix.shape[2]):
            for j in range(0,matrix.shape[3]):
                if np.abs(tempMax[i][j]) >= np.abs(tempMin[i][j]):
                    temp[i][j] = tempMax[i][j]
                else:
                    temp[i][j] = tempMin[i][j]
        result.append(temp)
    return np.asarray(result)

processTypeLoad = True

outfile = "saveFiles90Periods.npy"

if processTypeLoad == False:
    fileName,fileValue = readAllFile('dow30\*.csv')
    #correlationMatrix,adjustmentMatrix = calcAdjCorrMatrix(fileValue)
    a = calcAdjCorrMatrix(fileValue,540,1,"Standart")
    a = np.asarray(a).squeeze(axis=2)
    np.save(outfile,a)
    #rs = maxCorrPeriodic(a)

    # plt.imshow(pixelCorr[0][0], cmap='gray',norm=Normalize(vmin=0,vmax=255))
    # plt.show()
else:
    import math
    fileName, fileValue = readAllFile('dow30\*.csv')
    corr = np.load(outfile)
    pixelCorr = ((corr + 1) / 2) * 255
    print(len(fileValue))
    print(len(fileValue[0]))
    for i in range(0,240,30):
        a = []
        c = []
        # for k in range(30):
        #     b = fileValue[k][i*30:(i+1)*30]
        #     a.append(sum(b)/30.0)
        #     c.append(sum(abs(corr[i][0][k]))/30.0)
        # a = np.round(a, 2)
        # print(a[[14,17,21,24]])
        plt.figure()
        name = "Figure " + str(i)
        title_obj = plt.title(name)
        plt.setp(title_obj, color='r')
        plt.imshow(pixelCorr[i][0], cmap='gray', norm=Normalize(vmin=0, vmax=255))
    #print(a)
plt.show()
#x = np.array([[1,3,1,0],[3,1,2,5],[1,2,1,2],[0,5,2,1]])
#print(__calcCentralityVect(adjustmentMatrix,True))
#print(adjustmentMatrix)
#print(correlationMatrix)