import numpy as np
import pandas as pd
import glob

lastDaysAmount = 1000

def toCorrCoef(x,y,lastDaysAmount):
    coef = np.corrcoef(x[len(x) - lastDaysAmount:len(x) + 1], y[len(y) - lastDaysAmount:len(y) + 1])
    mean = np.mean(x[len(x) - lastDaysAmount:len(x) + 1])#to calculate mean of X array
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

def calcAdjCorrMatrix(x):
    correlationMatrix = np.zeros((len(fileValue), len(fileValue)))
    adjustmentMatrix = np.zeros((len(fileValue), len(fileValue)))
    for i in range(len(x)):
        for j in range(i+1):
            corrEf,mean = toCorrCoef(x[i],x[j], lastDaysAmount)
            distGraph = correToGraphLength(corrEf,mean)
            if i == j:
                adjustmentMatrix[i][j] = distGraph
                correlationMatrix[i][j] = corrEf
            else:
                adjustmentMatrix[i][j] = distGraph
                adjustmentMatrix[j][i] = distGraph
                correlationMatrix[i][j] = corrEf
                correlationMatrix[j][i] = corrEf
    return correlationMatrix,adjustmentMatrix

fileName,fileValue = readAllFile('stock\*.csv')
correlationMatrix,adjustmentMatrix = calcAdjCorrMatrix(fileValue)
#x = np.array([[1,3,1,0],[3,1,2,5],[1,2,1,2],[0,5,2,1]])
print(__calcCentralityVect(adjustmentMatrix,True))
print(adjustmentMatrix)
#print(correlationMatrix)