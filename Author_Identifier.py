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
    """A class that holds all the novel objects from a particular author"""
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.corpus = self.getCorpus(self.filepath)
    def getCorpus(self, filepath):
        """ Builds novel objects from all of the files in filepath and returns
        them in a list"""
        corpusList = []
        for directory,subdir,files in os.walk(filepath):
            for f in files:
                fh = open(directory + '/' + f, 'r')
                corpusList.append(Novel(fh.read(), f))
        return corpusList

class Novel:
    """A class that brings in text from a single file and analyzes it on initialization.
        The results of the analysis are stored in the dictionary summaryDict"""
    def __init__(self, text, filename):
        self.filename = filename
        self.fullText = text.split()
        #uses first 40k words of each novel
        self.text = " ".join(text.split()[:40000])
        self.wordTokenizedText = nltk.tokenize.word_tokenize(self.text)
        #frequency distribution
        self.fd = nltk.FreqDist(self.wordTokenizedText)
        self.numTokens = self.getNumTokens()
        self.summaryDict = {
                # getting these to be roughly 1>x>0.1 works best
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

####################################### MISC FUNCTION ######################################

def prime_data(listOfAuthors):
    """Big, ugly function to make values of summaryDict roughly btwn 0.1 and 1. It does this by
    multiplying or dividing all related values by some power of 10. This does not affect their relation
    to one another and improves performance significantly."""
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

#Creates author objects from wach subdirectory in the filepath. The "test" author is the
#one that contains all unidentified novels. 
allAuthors = []
for directory,subdir,files in os.walk(theFilepath):
    for s in subdir:
        authFilePath = theFilepath + "/" + s
        allAuthors.append(Author(s, authFilePath))
allAuthors = prime_data(allAuthors)

#Creates a separate list of all authors except the "test" author, which is
#assigned to the variable 'testAuth'.
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
