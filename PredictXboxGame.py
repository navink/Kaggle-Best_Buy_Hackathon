import xml.etree.ElementTree as ET
from VectorSpaces import VectorSpaces
from LatentSemanticIndexer import LSI
import csv
from collections import defaultdict
from correct import *
import sys

def clean( query ):
    query = re.sub( r'[\W]', '', query )
    query = query.lower()
    return query

def getProductNames(filename):
    productTree = ET.parse(filename)
    root = productTree.getroot()
    
    productNames = []
    productSku = defaultdict(list)
    
    for product in root.findall('product'):
        name = product.find('name').text
        productNames.append(name)
        productSku[name] = product.find('sku').text
        
    return productNames, productSku

def createQueryProductDictionary(filename):
    trainingFile = open(filename)
    reader = csv.reader(trainingFile)
    headers = reader.next()
    queryProdDict =  defaultdict(lambda: defaultdict(int))
    
    for line in reader:
        query = line[3]
        sku = line[1]
    
        query = clean(query)
        queryProdDict[query][sku] += 1
        
    return queryProdDict
            
if __name__ == "__main__":
    products, productSku = getProductNames('Data/small_product_data.xml')
    productVectors = VectorSpaces(products)
    productMatrix = productVectors.documentVectors    
    queryProdDict = createQueryProductDictionary('Data/train.csv')
    
    #test.csv
    test_file_name = 'Data/' + sys.argv[2]
    
    #Apply transformations, first TF-IDF before LSI
    lsi = LSI(productMatrix)    
    lsi.tfidfTransform()        
    lsi.lsiTransform()
    
    
    reader = csv.reader(test_file_name)
    headers = reader.next()

    #results.csv
    output_file_name = 'Data/' + sys.argv[1]
    outputFile = open(output_file_name, 'wb')
    writer = csv.writer(outputFile, delimiter = " " )
    
    queries = queryProdDict.keys()
    numqueries = {}
    
    for query in queries:
        numqueries[query] = 1
    
    for line in reader:
        skus = []
        query = line[2]
        orig_query = query
        query = clean(query)
        
        if query in queryProdDict:
            for sku in sorted( queryProdDict[query], key=queryProdDict[query].get, reverse = True ):
                skus.append( sku )
            
            skus = skus[0:5]
        
        # query spelling correction    
        if len( skus ) < 5:
            if len( query ) < 6:
                corrected_queries = edits1( query )
            else:
                corrected_queries = correct( numqueries, query )

            corrected_found = [x for x in corrected_queries if x in queryProdDict and x != query]
            
            if corrected_found:
                skus_counts = {}
                
                for c_query in corrected_found:
                    skus_counts.update( queryProdDict[c_query] )

                additional_skus = sorted( skus_counts, key=skus_counts.get, reverse = True )
                
                for sku in skus:
                    if sku in additional_skus:
                        additional_skus.remove(sku)
                    
                skus.extend( additional_skus )
                skus = skus[0:5]
                
        # search in product names
        if len( skus ) < 5:
            productVectors.documentVectors = lsi.getLSIMatrix()            
            productIndices = productVectors.search([orig_query])
            skus_from_xml = [productSku[products[productIndex]] for productIndex in productIndices if productSku[products[productIndex]] not in skus]
            skus.extend( skus_from_xml )            
            skus = skus[0:5]
            
        writer.writerow( skus )    
    
    print "Predictions have been written to the output file."        