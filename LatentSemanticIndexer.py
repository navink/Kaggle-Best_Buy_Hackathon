from scipy import linalg,array,dot,mat
from math import log
from scipy import zeros
import numpy as np

class LSI:
	""" Latent semantic indexing(LSI). 
	    Apply transforms to a document-term matrix to bring out latent relationships. 
	    It is an indexing and retrieval method that uses a mathematical technique called Singular value decomposition (SVD) 
	    to identify patterns in the relationships between the terms and concepts contained in an unstructured collection of text.
	"""


	def __init__(self, matrix):
		self.matrix = array(matrix)


	def __repr__(self,):
		""" Make the matrix look pretty """
		stringRepresentation=""

		rows,cols = self.matrix.shape

		for row in xrange(0,rows):
			stringRepresentation += "["

			for col in xrange(0,cols):
				stringRepresentation+= "%+0.2f "%self.matrix[row][col]
			stringRepresentation += "]\n"

		return stringRepresentation
		

	def __getTermDocumentOccurences(self,col):
		""" Find how many documents a term occurs in"""

		termDocumentOccurences=0
		
		rows,cols = self.matrix.shape

		for n in xrange(0,rows):
			if self.matrix[n][col]>0: #Term appears in document
				termDocumentOccurences+=1 
		return termDocumentOccurences


	def tfidfTransform(self,):	
		""" Expects a bag-of-words (integer values) training corpus during initialization. During transformation, it will 
		    take a vector and return another vector of the same dimensionality, except that features which were rare in the 
		    training corpus will have their value increased.
	   	    
		    With a document-term matrix: matrix[x][y]
			tf[x][y] = frequency of term y in document x / frequency of all terms in document x
			idf[x][y] = log( abs(total number of documents in corpus) / abs(number of documents with term y)  )		    
		"""

		documentTotal = len(self.matrix)
		rows,cols = self.matrix.shape

		for row in xrange(0, rows): #For each document
		   
			wordTotal= reduce(lambda x, y: x+y, self.matrix[row] )

			for col in xrange(0,cols): #For each term
			
				#For consistency ensure all self.matrix values are floats
				self.matrix[row][col] = float(self.matrix[row][col])

				if self.matrix[row][col]!=0:

					termDocumentOccurences = self.__getTermDocumentOccurences(col)

					termFrequency = self.matrix[row][col] / float(wordTotal)
					inverseDocumentFrequency = log(abs(documentTotal / float(termDocumentOccurences)))
					self.matrix[row][col]=termFrequency*inverseDocumentFrequency


	def lsiTransform(self,dimensions=1):
		""" Calculate SVD of objects matrix: U . SIGMA . VT = MATRIX 
		    Reduce the dimension of sigma by specified factor producing sigma'. 
		    Then dot product the matrices:  U . SIGMA' . VT = MATRIX'
		"""
		rows,cols= self.matrix.shape

		if dimensions <= rows: #Its a valid reduction			
			u,sigma,vt = linalg.svd(self.matrix, full_matrices=False)
			
			#Dimension reduction, build SIGMA'			
			sigmaprime = np.diag(sigma)			
			print np.allclose(self.matrix, np.dot(u, np.dot(sigmaprime, vt)))			
			reconstructedMatrix = np.dot(u, np.dot(sigmaprime, vt))
			
			#Save transform
			self.matrix=reconstructedMatrix

		else:
			print "dimension reduction cannot be greater than %s" % rows

	def getLSIMatrix(self):
		modMatrix = zeros((len(self.matrix), len(self.matrix[0])), dtype=float)
		
		for i in range(len(self.matrix)):
			for j in range(len(self.matrix[0])):
				modMatrix[i][j] = float("{0:.2f}".format(self.matrix[i][j]))
		
		return modMatrix		
		
