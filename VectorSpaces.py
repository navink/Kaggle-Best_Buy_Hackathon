from Parser import Parser
from sets import Set
from numpy import dot
from numpy.linalg import norm
	
class VectorSpaces:
	""" A algebraic model for representing text documents as vectors of identifiers. 
	A document is represented as a vector. Each dimension of the vector corresponds to a 
	separate term. If a term occurs in the document, then the value in the vector is non-zero.
	"""

	#Collection of document term vectors
	documentVectors = []

	#Mapping of vector index to keyword
	vectorKeywordIndex=[]

	parser=None


	def __init__(self, documents=[]):
		self.documentVectors=[]
		self.parser = Parser()
		if(len(documents)>0):
			self.build(documents)

	def removeDuplicates(self, list):		
		return Set((item for item in list))

	def cosineSimilarity(self, vector1, vector2):
		""" Calculate Cosine Similarity between the two document vectors :
			cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
		return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

	def build(self,documents):
		""" Create the vector space for the input document """
		self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)

		self.documentVectors = [self.createVector(document) for document in documents]


	def getVectorKeywordIndex(self, documentList):
		""" create the keyword associated to the position of the elements within the document vectors """

		#Mapped documents into a single word string	
		vocabularyString = " ".join(documentList)
			
		vocabularyList = self.parser.tokenise(vocabularyString)
		#Remove common stop words which have no search value
		vocabularyList = self.parser.removeStopWords(vocabularyList)
		uniqueVocabularyList = self.removeDuplicates(vocabularyList)
		
		vectorIndex={}
		offset=0
		#Associate a position with the keywords which maps to the dimension on the vector used to represent this word
		for word in uniqueVocabularyList:
			vectorIndex[word]=offset
			offset+=1
		return vectorIndex  #(keyword:position)


	def createVector(self, wordString):
		""" @pre: unique(vectorIndex) """

		#Initialize vector with 0's
		vector = [0.0] * len(self.vectorKeywordIndex)
		wordList = self.parser.tokenise(wordString)
		wordList = self.parser.removeStopWords(wordList)
		for word in wordList:			
			if word in self.vectorKeywordIndex:
				vector[self.vectorKeywordIndex[word]] += 1.0; #Use simple Term Count Model (Bag of words)
		return vector


	def buildQueryVector(self, termList):
		""" convert query string into a term vector """
		query = self.createVector(" ".join(termList))
		return query


	def search(self,searchList):
		""" search for documents that match based on a list of terms from a query"""
		queryVector = self.buildQueryVector(searchList)
		
		rankings = [self.cosineSimilarity(queryVector, documentVector) for documentVector in self.documentVectors]
		rankIndices = [i[0] for i in sorted(enumerate(rankings), key = lambda x:x[1], reverse=True)]
		
		return rankIndices
