# Author: Everett Stenberg (Steinshark)
import zlib 
import numpy
import xxhash
from collections import Counter 
import os 

#todo: remove the top-k tokens as stopwords since they add little info (i dont care about 'the')
stopwords       = [] 
EOT_TOKEN       = "<|endoftext|>"
#Subject to change 
hash_fn         = xxhash.xx3_64
hash_len        = 64

def clean_doc(contents:str):
    #Lower 
    contents    = contents.lower()
    
    #Remove whitespace idiocy
    while "  " in contents:
        contents = contents.replace("  ", " ")
    while "\n\n" in working_text:
        contents = contents.replace("\n\n", "\n")

    return contents
    

    
def word_doc_counter(ds_path:str):
    
    term_freqs      = {}
    n_docs          = 0
    
    for document in os.listdir(ds_path):
        contents         = open(document,'r').read()

        documents        = contents.split(EOT_TOKEN)

        for doc in documents:
            contents         = clean_doc(contents)
            words            = set(contents.split(" "))
            n_docs           += 1
            for word in words:
                term_freqs[word] += 1
    
    return term_freqs, n_docs

def tf_idf(term,term_count,n_terms,n_docs):
    tf                 = term_count / n_terms 
    idf                = numpy.log(word_occurences[term] / n_docs)

return tf * idf
    
    
#This function processess a single document for SimHash analysis 
def SimHash_doc(doc:str,n_docs):
    
    #Tokenize
    doc_tokens      = tokenize_document(doc,n_docs)
    
    #Hash 
    document_hash   = hash_document(doc_tokens)
    return document_hash


#Convert the hash (intdigest form) into a bitarray
def bit_arr(hash_digest:int):
    bit_arr     = [0 for _ in range(hash_len)]
    for i in range(hash_len):
        bit_arr[i] = (hash_digest >> i) & 1 
    
    return numpy.asarray(bit_arr) * 2 - 1 
    
    
#This function cononacalizes the contents of a document
#and returns an array of scaled tokens based on word count (log count) 
def tokenize_document(doc:str,n_docs):
    
    #Lower 
    working_text    = clean_doc(doc)
    
    #Split and count words 
    working_words   = working_text.split(" ")
    word_counts     = Counter(working_words)
    log_word_counts = {word: tf_idf(word,count,len(working_words),n_docs) for word,count in word_counts.items()}
    
    #Get tokens
    encoded_words   = [word.encode() for word in working_words]
    tokens          = [bit_arr(word.intdigest()) for word in map(hash_fn,encoded_words)]
    
    #Scale each token by log TF
    weighted_tokens = []
    for token, word in zip(tokens,working_words):
        #Skip words that add little value
        if word in stopwords:
            continue
        weighted_token = token * log_word_counts[word]
        weighted_tokens.append(weighted_token)
    
    return weighted_tokens


#This function hashes the document based on the weighted tokens 
def hash_document(weighted_tokens:numpy.array):
    
    #Accumulate weights of document based on each token's contribution
    accu_vector = numpy.zeros((hash_len,))
    for token in weighted_tokens:
        accu_vector += token

    #Determine document hash
    document_hash = numpy.zeros((hash_len,))
    for i in range(len(accu_vector)):
        document_hash[i] = int(accu_vector[i] > 1)
    
    return document_hash

if __name__ == "__main__":
    counts,n_docs = word_doc_counter()
    document = None
    tokenized = SimHash_doc(document,n_docs)
    print(tokenized)
 
