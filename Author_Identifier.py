#!/usr/bin/env python
"""
LING 131A Final Project
@author: Jackson Breyer
Fall/Winter 2014
"""
import nltk, sys, os
from nltk.classify import maxent

############################################################################
#                   ENTER FILEPATH HERE                                    #
############################################################################

theFilepath = "ENTER CORPUS FILEPATH HERE"

############################################################################
#                                                                          #
############################################################################

class Author:
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.corpus = self.getCorpus(self.filepath)
    def getCorpus(self, filepath):
        corpusList = []
        for directory,subdir,files in os.walk(filepath):
            for f in files:
                fh = open(directory + '/' + f, 'r')
                corpusList.append(Novel(fh.read(), f))
        return corpusList

class Novel:
    def __init__(self, text, filename):
        self.filename = filename
        self.fullText = text.split()
        #uses first 40k words of each novel
        self.text = " ".join(text.split()[:40000])
        self.wordTokenizedText = nltk.tokenize.word_tokenize(self.text)
        self.fd = nltk.FreqDist(self.wordTokenizedText)
        self.numTokens = self.getNumTokens()
        self.summaryDict = {
                # for some reason, getting these to be around 1>x>0.1 works best
                # see prime_data().
                "Word Count":               self.wordCount(),
                "Frequency of And":         self.freqOfAnd(),
                "Frequency of Dialogue Tag":self.freqOfDialogue(),
                "Frequency of Upper Case":  self.freqOfUpperCase(),
                "Freq of MFW1-MFW2":        self.freqOfMFWXMinusMFWY(1,2),
                "Freq of MFW2-MFW3":        self.freqOfMFWXMinusMFWY(2,3),
                "Freq of MFW3-MFW4":        self.freqOfMFWXMinusMFWY(3,4),
                "Freq of MFW1-MFW3":        self.freqOfMFWXMinusMFWY(1,3),
                "Freq of MFW2-MFW4":        self.freqOfMFWXMinusMFWY(2,4)
                }
    def getNumTokens(self):
        numTokens = 0
        for key in self.fd.keys():
            numTokens += self.fd[key]
        return numTokens

    # STYLOMETRIC FEATURE CALCULATIONS
    def wordCount(self):
        return len(self.fullText)
    def freqOfDialogue(self):
        return ((self.fd["''"]+self.fd["``"]+self.fd["\""]+self.fd["'"])/self.numTokens)
    def freqOfAnd(self):
        return (self.fd["and"]/self.numTokens)
    def freqOfUpperCase(self):
        count = 0
        for key in self.fd.keys():
            if key[0].upper() == key[0]:
                count += 1
        return (count/self.numTokens)
    #Frequency of the xth most frequent word minus the frequency of the yth most frequent word
    def freqOfMFWXMinusMFWY(self, x, y):
        return (self.fd.most_common(y)[x-1][1]/self.numTokens)-(self.fd.most_common(y)[y-1][1]/self.numTokens)

# Function to make values of summaryDict roughly btwn 0.1 and 1. This improves performance significantly.
def prime_data(listOfAuthors):
    for feature in listOfAuthors[0].corpus[0].summaryDict.keys():
        allOfFeature = []
        for author in listOfAuthors:
            for novel in author.corpus:
                allOfFeature.append(novel.summaryDict[feature])
        biggest = max(allOfFeature)
        multp = 0
        while biggest >= 1:
            biggest = biggest/10
            multp+=1
        if multp > 0:
            for author in listOfAuthors:
                for novel in author.corpus:
                    novel.summaryDict[feature] = novel.summaryDict[feature]/(10**multp)
        multp = 0
        while biggest < 0.1:
            biggest = biggest*10
            multp+=1
        if multp > 0:
            for author in listOfAuthors:
                for novel in author.corpus:
                    novel.summaryDict[feature] = novel.summaryDict[feature]*(10**multp)
    return listOfAuthors
    
####################################### CREATE AUTHORS ######################################

allAuthors = []
for directory,subdir,files in os.walk(theFilepath):
    for s in subdir:
        authFilePath = theFilepath + "/" + s
        allAuthors.append(Author(s, authFilePath))
allAuthors = prime_data(allAuthors)

knownAuthors = []
for auth in allAuthors:
    if auth.name.lower() != "test":
        knownAuthors.append(auth)
    else:
        testAuth = auth

######################################## RUN TEST ############################################

train = [(author.corpus[x].summaryDict, author.name) for author in knownAuthors for x in range(0,len(author.corpus))]
test = [(testAuth.corpus[x].summaryDict) for x in range(0,len(testAuth.corpus))]
encoding = maxent.TypedMaxentFeatureEncoding.train(train, count_cutoff=3, alwayson_features=True)
classifier = maxent.MaxentClassifier.train(train, bernoulli=False, encoding=encoding)
result = classifier.classify_many(test)

######################################## PRINT RESULTS #####################################
for i in range(0,len(testAuth.corpus)):
    print("\nPrediction for " + testAuth.corpus[i].filename + ":")
    print(result[i])
