Jackson Breyer

Fall/Winter, 2014

Determining Authorship Through Stylometric Classification

LANGUAGE:

	Python 3
	
EXTERNAL MODULES / PACKAGES REQUIRED:

	NLTK - Natural Language Toolkit
	NUMPY - Numerical Python
	
OVERVIEW:

	For each of four authors, this program reads in several novels written by that author. 
	It calculates statistics about the writing style of each text, in an attempt to fingerprint
	the author's writing style. Then the program uses a Maximum Entropy Classifier to identify 
	the author of previously unseen novels, written by one of the four previously seen authors. 
	
CLASSES / STRUCTURES:

	Novel:
		The Novel class serves as a corpus reader for each file that is read in. It houses a 
		few versions of the text and related data. That data includes the author and various 
		statistics about the text's style. Those statistics are calculated on initialization. 
		Note that most of the statistics are calculated based on the first 40,000 words of the
		text, since I perceive that to be a representative sample. Keeping the sample sizes 
		consistent makes the statistics relevant for comparison with other novels. The statistics
		used for comparison are stored as {stat name : value} in the dictionary summaryDict.
		
	Author:
		The Author class serves to organize all the texts from a particular author. Each Author 
		object takes an author name and a filepath as arguments. On initialization, the object
		will create Novel objects from the files in the filepath given and all of its subfolders. 
		Then it stores all of those novel objects in a list named 'corpus'. 

PROGRAM FLOW:

	1. User Enters Information:
		In order to run this program, the user must enter the name of each author (4 in total) and
		a filepath that holds text files of novels written by the associated author. Finally, the 
		user must enter a filepath for the test files. This path must contain some number of text 
		files written by the four authors. These need not all be written by the same author, but 
		they must be written by one of the four authors entered earlier. 

	2. Create Authors:
		The program begins by creating the Author objects for each of the four authors entered. 
		These build Novel objects from the files in their given filepath. The Author objects are 
		stored inthe knownAuthors list. Additionally, the program creates an author from the test 
		files, which is stored in the variable testAuth. This is not stored in knownAuthors. All 
		authors, known and unknown, are put through the prime_data() function together. This 
		function multiplies or divides the items in summaryDict by a power of 10 so that they fall 
		roughly between 0.1 and 1. Each feature, across all authors, is multiplied/divided by the 
		same power of 10 so the relative proportions stay consistent. I've found that performing 
		these multiplications/divisions significantly increases the performance of this program. 

	3. Run the Maxent Classifier:
		Training and test sets are defined from the Authors that have been created. The named authors 
		become the training data and the unknown author becomes the test set. A classifier object is 
		built from the training data, and then it is used to identify the test data. 

	4. Print Results:
		Finally, the results are printed for each test file. For each of those, the program gives the 
		filename and then a guess for that file's author.
