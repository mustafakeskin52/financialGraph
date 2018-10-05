import numpy as np
import pandas as pd
import glob

lastDaysAmount = 30

def toCorrCoef(x,y,lastDaysAmount,xlastBoundary,ylastBoundary):

    coef = np.corrcoef(x[xlastBoundary - lastDaysAmount:xlastBoundary], y[ylastBoundary - lastDaysAmount:ylastBoundary])
    mean = np.mean(x[xlastBoundary - lastDaysAmount:xlastBoundary])#to calculate mean of X array
    return coef[1,0],mean
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

def calcAdjCorrMatrix(x,maxshifting,maxshiftingAmount):
    timeSeriesList = []
    #the first parameter of timeSeriesList is given by input to indicate time that outside want to receive.
    #the second parameter of timeSeriesList is given by input to show shifting corelation matrix
    #If maxshiftingAmount is zero,the list won't return second dimension array
    for t in range(0,maxshifting,1):
        timeSeriesList.append([])
        for shiftingAmount in range(0,maxshiftingAmount,1):
            timeSeriesList[t].append([])
            correlationMatrix = np.zeros((len(fileValue), len(fileValue)))
            adjustmentMatrix = np.zeros((len(fileValue), len(fileValue)))
            for i in range(len(x)):
                for j in range(len(x)):
                    lastXBoundary = len(x[i])-t - shiftingAmount
                    lastYBoundary = len(x[j])-t
                    corrEf,mean = toCorrCoef(x[i],x[j], lastDaysAmount,lastXBoundary,lastYBoundary)
                    distGraph = correToGraphLength(corrEf,mean)
                    adjustmentMatrix[i][j] = distGraph
                    correlationMatrix[i][j] = corrEf
            timeSeriesList[t][shiftingAmount].append(correlationMatrix)
    return timeSeriesList
fileName,fileValue = readAllFile('dow30\*.csv')
#correlationMatrix,adjustmentMatrix = calcAdjCorrMatrix(fileValue)
#a = timeseriesCorrMatrix(fileValue,1000)
a = calcAdjCorrMatrix(fileValue,100,2)
print(a[0])
#x = np.array([[1,3,1,0],[3,1,2,5],[1,2,1,2],[0,5,2,1]])
#print(__calcCentralityVect(adjustmentMatrix,True))
#print(adjustmentMatrix)
#print(correlationMatrix)