import sys
import numpy as np
 
def main():
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    # read File
    dataset = loadFile(inFile)
    cols = len(dataset[0])
    rows = len(dataset)
    # stripping the labels of each column
    dataset = dataset [1:rows,:]
    labels = dataset[:,cols-1]
    cImp = classImpurity(labels)
    error = findError(labels)
    file = open(outFile,"w+")
    file.write("gini_impurity: %f\n" %cImp)
    file.write("error: %f\n" %error)
   
 
 
 
def loadFile (inputFile):
    matrix =[]
    file = open(inputFile)
    for line in file:
        line = line.rstrip('\r\n')
        line = line.rstrip('\n')
        val = line.split('\t')
        matrix.append(val)
    return np.array(matrix)
 
 
 
 
def classImpurity (array):
    total = len(array)
    c1 = array[0]
    count1 = 0
    count2 = 0
    for value in array:
        if value == c1:
            count1 = count1 + 1
        else:
            count2 = count2 + 1
    return (count1/total)*(1-(count1/total))+(count2/total)*(1-(count2/total))
 
 
def findError (label):
    c1=0
    c2=0
    class1 = label[0]
    for value in label:
       if (value==class1):
           c1=c1+1
       else:
           c2=c2+1
    return min(c1,c2)/len(label)
 
 
main()
