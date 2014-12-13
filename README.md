Determining Authorship Through Classification
==================================
LANGUAGE:

	Python 3
	
EXTERNAL MODULES / PACKAGES REQUIRED:

	NLTK - Natural Language Toolkit
	NUMPY - Numerical Python

GOAL:

	 Determine the author of test novels based on a training set of novels with their authors 
	 already identified.
	
OVERVIEW:

	For each subdirectory (author) in an input filepath, this program reads in several novels 
	written by that author. A subdirectory named test is used as test data. The program
	calculates statistics about the writing style of each author's texts, in an attempt to 
	fingerprint the author's writing style. Then, the program uses a Maximum Entropy Classifier 
	to identify the author of the test data--previously unseen novels which are presumed to have 
	been written by one of the previously seen authors. Please note that the text files for all 
	novels in this repository are taken from https://www.gutenberg.org/
	
CLASSES / STRUCTURES:

	Novel:
		The Novel class serves as a corpus reader for each file that is read in. It houses
		a few versions of the text and related data. That data includes the author and 
		various statistics about the text's style. Those statistics are calculated on 
		initialization. Note that most of the statistics are calculated based on the first 
		40,000 words of the text, since I perceive that to be a representative sample. 
		Keeping the sample sizes consistent also makes the statistics relevant for comparison 
		with other novels. The statisticsused for comparison are stored as {stat name : value} 
		in the dictionary summaryDict.
		
	Author:
		The Author class serves to organize all the texts from a particular author. Each 
		Author object takes an author name and a filepath as arguments. In prsactice, the 
		name argument is derived from the subdirectory name that holds the author's texts. 
		On initialization, the Author object will create Novel objects from the files in the 
		filepath given and all of its subfolders. Then it stores all of those novel objects 
		in a list named 'corpus'. 

PROGRAM FLOW:

	1. User Enters Information:
		In order to run this program, the user must enter a corpus filepath. In the filepath, 
		there must be one subdirectory per known author. Each subdirectory must contain some 
		number of novels by that author. There must also be a subdirectory named "Test", which 
		will house the novels to be evaluated. These need not all be written by the same author, 
		but they should each be written by one of the known authors. 

	2. Create Authors:
		The program begins by creating the Author objects for each of the author filepaths. 
		These Author objects build Novel objects from the files in their respective filepaths. 
		The Author objects are all stored in a list called allAuthors. This list is processed 
		by the prime_data function, which multiplies or divides the items in summaryDict by a 
		power of 10 so that they fall roughly between 0.1 and 1. Each feature, across all 
		authors, is multiplied/divided by the same power of 10 so the relative proportions stay
		consistent. I've found that performing these multiplications/divisions significantly 
		increases the performance of this program. Finally, the allAuthors is separated into a 
		knownAuthors list and a variable testAuth, which holds the test Author. 

	3. Run the Maxent Classifier:
		Training and test sets are defined from the Authors that have been created. The 
		knownAuthors become the training data and the testAuth becomes the test set. A 
		classifier object is built from the training data, and then it is used to identify 
		the test data. 

	4. Print Results:
		Finally, the results are printed for each test file. For each of those, the program 
		gives the filename and then a guess for that file's author.
