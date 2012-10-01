from VectorSpaces import VectorSpaces
from LatentSemanticIndexer import LSI

if __name__ == '__main__':

    documents = ["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."]
    docVectors = VectorSpaces(documents)
    docMatrix = docVectors.documentVectors
    
    lsi = LSI(docMatrix)    
    print lsi
    
    lsi.tfidfTransform()
        
    lsi.lsiTransform()
    print lsi.getLSIMatrix()
    
    query = "cats and dogs"
    print docVectors.search([query])