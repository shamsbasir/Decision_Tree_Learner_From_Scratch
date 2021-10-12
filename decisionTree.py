import numpy as np
import sys
#:%s/search/replace/g
class Node:
	def __init__(self, mydata, myLevel, dmax, featureList, usedFeatureList,labelType,myPrintTag=""):
		# my data 
		self.mydata		= mydata
		# my depth or level in the tree
		self.myLevel		= myLevel
		# maximum depth
		self.dmax		= dmax
		# all the feature list
		self.featureList	= featureList
		# used feature list
		self.usedFeatureList	= usedFeatureList
		# splitting feature 
		self.splittingIndex	= None
		# assigning the leaf
		self.isLeaf		= False
		# values yes/no for example 
		self.splittingIndexVal  = None
		# children 
		self.childrens		= []
		# original label types 
		self.labelTypes		= labelType
		# counts of labels for this node
		self.myLabelCount	= self.whatIsMyLabelCount(mydata[:,-1],self.labelTypes)
		# majority vote
		self.majorityVote	= self.majorityVote(self.myLabelCount)
		# printing tag
		self.myPrintTag		= myPrintTag

     # counting the majority vote: data is the labels of the current node
	def majorityVote(self,myLabelCount):
		Vals = list(myLabelCount.values())

		Keys = [*myLabelCount]

		if Vals[0] > Vals[1]:

			candidate= Keys[0]

		elif Vals[1] >Vals[0]:

			candidate= Keys[1]

		elif Vals[0] == Vals[1]:
			#if Keys[0] > Keys[1]:
	
			#	candidate = Keys[0]	
			#else:
			#	candidate = Keys[1]	
			candidate = max(Keys)

		return candidate


	   # this function creates printing tag
	def createPrintTag(self,LocalLabelCount):
		LabelList = list(LocalLabelCount.keys())
		LabelVal  = list(LocalLabelCount.values())
		#print("Create Tag ",LabelList,LabelVal)	
		#if len(LabelList)==2:
		message = "[%d %s / %d %s]"%(LabelVal[0],LabelList[0],LabelVal[1],LabelList[1])
		#else:
		 #   message = "[%d %s]"%(LabelVal[0],LabelList[0])
		#print("create Tag : ",message)
		return message

	def whatIsMyLabelCount(self,myLabels,LabelTypes):
                # print("What is my Label Count : LabelTypes ",LabelTypes[0])
                # Make a map of Labels and their respective counts
                LocalLabelCount = {LabelTypes[0]:0,LabelTypes[1]:0}
                # grab the data 
                for Label in myLabels:
                                #print("label =",Label)
                                # we have the LocalLabelCount filled with zeros 
                                LocalLabelCount[Label] += 1

                return LocalLabelCount

       # marginal gini impurity : send the labels of the current node  
	def classGiniImpurity(self,array):
		# in this function : we give an array of labels,
		# it outputs the gini impurity: it can be applied to subclasses too
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


	def childrenGiniImpurity(self,featureLabel):
		# first sort out of labels for yes, no and then calc calcGini
		featureVals	= np.unique(featureLabel[:,0])
		if len(featureVals) == 2:
			child_one	= featureVals[0]
			child_two	= featureVals[1]
			data_child_one	= []
			data_child_two	= []
			child_one_count = 0
			child_two_count = 0
		else:
			child_one	= featureVals[0]
			child_two	= None
			data_child_one	= []
			data_child_two	= []
			child_one_count = 0
			child_two_count = 0
		
		total,col	= featureLabel.shape
		conditionalGiniImpurity = 0
		# grab data for child_one and child_two:
		for val,label in featureLabel:
			if val == child_one: 
				data_child_one.append(label)
				child_one_count +=1
			if val == child_two:
				data_child_two.append(label)
				child_two_count +=1

		if len(featureVals) == 2:
			data_child_one = np.array(data_child_one)
			data_child_two = np.array(data_child_two)
			# now calculate the gini impurity of child_one and child_two
			# these are the conditional gini impurities
			child_one_gini_impurity = self.classGiniImpurity(data_child_one)
			child_two_gini_impurity = self.classGiniImpurity(data_child_two)
			# gini impurity of both children are calculated
			child_one_gini =child_one_gini_impurity*child_one_count/total
			child_two_gini =child_two_gini_impurity*child_two_count/total
			# total gini impurity of the feature
			conditionalGiniImpurity = child_one_gini+child_two_gini
		else:

			data_child_one = np.array(data_child_one)
			#data_child_two = np.array(data_child_two)
			# now calculate the gini impurity of child_one and child_two
			# these are the conditional gini impurities
			child_one_gini_impurity = self.classGiniImpurity(data_child_one)
			#child_two_gini_impurity = self.classGiniImpurity(data_child_two)
			# gini impurity of both children are calculated
			child_one_gini =child_one_gini_impurity*child_one_count/total
			#child_two_gini =child_two_gini_impurity*child_two_count/total
			# total gini impurity of the feature
			conditionalGiniImpurity = child_one_gini #+child_two_gini
	
		return conditionalGiniImpurity



	def branchOff(self,featureList,usedFeatureList,labelTypes):
		parentNodeGiniImpurity= self.classGiniImpurity(self.mydata[:,-1])	
		GiniGain = -200
		bestSplittingIndex = -20
		
		for index in range(len(featureList)):
			# make sure the feature is not visited
			if featureList[index] not in usedFeatureList:
				# find the impurity of the feature 
				childrenGiniImpurity=self.childrenGiniImpurity(self.mydata[:,[index,-1]])
				# check the information gain 
				if parentNodeGiniImpurity - childrenGiniImpurity > GiniGain:
					bestSplittingIndex = index
					GiniGain = parentNodeGiniImpurity - childrenGiniImpurity
		 # here save the next best splitting feature and its value						 
		self.splittingIndex = featureList[bestSplittingIndex]

		# could use      
		self.splittingIndexVal = list(set(self.mydata[:, bestSplittingIndex]))

		for value in self.splittingIndexVal:
			# for y/n values 
			childrenData = []
			for data in self.mydata:
					# get the training data for children
				if data[bestSplittingIndex] == value:
					childrenData.append(data)
			childrenData = np.array(childrenData)
			# tag 
			myPrintTag = "{} = {}: ".format(self.splittingIndex, value)
			# recursively creating nodes 
			self.childrens.append(Node(childrenData, self.myLevel + 1, self.dmax,
			self.featureList, self.usedFeatureList + [self.splittingIndex],labelTypes,myPrintTag))

		return None



	# train function 
	def train(self,labelTypes):
		if self.myLevel == self.dmax:
				self.isLeaf = True
				return
			# if no attribute left 
		if len(self.featureList) == len(self.usedFeatureList):
				self.isLeaf =True
				return
	
		parentNodeGiniImpurity= self.classGiniImpurity(self.mydata[:,-1])	

		if parentNodeGiniImpurity == 0:
				self.isLeaf = True
				return
		else:
			self.branchOff(self.featureList,self.usedFeatureList,labelTypes) 

		for children in self.childrens:

			children.train(labelTypes)
		return


	def predict(self, thisFeature,annotate = False):
			 # it has 'A':'n','B':'y'
			#print("thisFeature ",thisFeature)
			# leaf reached 
			if self.isLeaf == True:
					return self.majorityVote
			# grabbing 
			thisFeatureVal = thisFeature[self.splittingIndex]
			#print(" in predict attr_value",attr_value)

			for spInVal in range(len(self.splittingIndexVal)):
				if self.splittingIndexVal[spInVal] == thisFeatureVal:
					return self.childrens[spInVal].predict(thisFeature)
			return self.majorityVote


	def print(self):
			print("{}".format("| " * self.myLevel + self.myPrintTag+self.createPrintTag(self.myLabelCount)))
			for children in self.childrens:
					children.print()
			return
			

class DecisionTree:
		def __init__(self, mydata_file, dmax,annotate=False):
			#self.mydata, self.attributes, self.label = self.parse_mydata_file(mydata_file)
			
			self.mydata, self.attributes, self.label = self.loadFile(mydata_file)
			self.labelType = np.unique(self.mydata[:,-1])
			self.root = Node(self.mydata, 0, dmax, self.attributes, [],self.labelType)



		def loadFile(self,infile):
			data		= []
			targetName	= []	
			featureName	= []
			dataset		= np.loadtxt(infile,dtype=str,delimiter='\t',skiprows=0)
			data		= dataset[1:,:]
			targetName	= dataset[0,-1]
			featureName	= dataset[0,0:-1]
			return data,featureName,targetName

		def train(self):
				self.root.train(self.labelType)

		def predict(self, thisFeature,annotate):
				return self.root.predict(thisFeature,annotate)

		def sketch(self):
				self.root.print()




class postProcessingTree:

	def __init__(self,tree,trnInfile,tstInfile,trnLbl,tstLbl,mtrc,annotate=False):
			
		file = open(trnLbl, "w")
		misClassifiedCount = 0
		featureLog, labelLog = self.loadFile(trnInfile)
		for label, thisFeature in zip(labelLog, featureLog):
				pred = tree.predict(thisFeature,annotate)
				if pred != label:
						misClassifiedCount += 1
				file.write("{}\n".format(pred))
		file.close()
		trainError = misClassifiedCount / len(labelLog)

		file = open(tstLbl, "w")



		misClassifiedCount = 0
		featureLog, labelLog = self.loadFile(tstInfile)
	   
		for label, thisFeature in zip(labelLog, featureLog):
				pred = tree.predict(thisFeature,annotate)
				#print("\n",thisFeature)
				if pred != label:
						misClassifiedCount += 1
				file.write("{}\n".format(pred))
		file.close()
		testError = misClassifiedCount / len(labelLog)

		file = open(mtrc, "w")
		file.write("error(train): {}\n".format(trainError))
		file.write("error(test): {}\n".format(testError))
		file.close()



	def loadFile(self,infile):
		mydata	= []
		label		= []	
		attributes	= []
		dataset		= np.loadtxt(infile,dtype=str,delimiter='\t',skiprows=0)
		mydata	= dataset[1:,:]
		label		= dataset[0,-1]
		attributes	= dataset[0,0:-1]
		featureLog = []
		labelLog = []
		for data in mydata:
				thisFeature = {}
				for i in range(len(attributes)):
						thisFeature[attributes[i]] = data[i]
				featureLog.append(thisFeature)
				labelLog.append(data[-1])
		return featureLog, labelLog
	

def main():
		trnInfile = sys.argv[1]
		tstInfile = sys.argv[2]
		dmax = int(sys.argv[3])
		trnLbl = sys.argv[4]
		tstLbl = sys.argv[5]
		mtrc = sys.argv[6]
		dmax = [dmax]
		#print("dmax = ",dmax)
		#dmax = [1,2,3,4,5,6,7]
		for dmx in dmax:
			# decision tree
			tree = DecisionTree(trnInfile, dmx)
			tree.train()
			tree.sketch()
			#calculating erros 
			postProcessingTree(tree,trnInfile,tstInfile,trnLbl,tstLbl,mtrc,annotate=True)
			
if __name__ == "__main__":
		main()






