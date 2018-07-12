import nltk
import math
import pandas

from collections import Counter
from unidecode import unidecode
from typing import List, Dict, Set

DISJUNCTION = 'OR'
FIRST_WORD_INDEX = 0
QUERY_INDEX = 1
SECOND_WORD_INDEX = 2

def load_data_from_cvs(path: str) -> pandas.DataFrame:
    return pandas.read_csv(path)


# lambda funcion to normalize text to lower case.
normalize = lambda text: unidecode(text.lower())


# lambda function to split text in tokens.
tokenize = lambda row: row.split()


# lambda function to summarize frequence of token in a article.
counter = lambda row: Counter(row)

def count_frequence(article: str, token: str) -> Counter:
    """Count frequence of token in a specified article.
    :param article: Contente of a article as a string.
    :param token: Token that frequence will be counted.
    :returns: Frequence of token in article.
    """
    counter = Counter(article)
    return counter[token]

def idf(inverted_index: List, token: str, collection_size: int) -> float:
    """Calc the idf of a token in index.
    :param inverted_index: Index that will be used for get term frequence
                           of a token.
    :param token: Token that will be calc idf.
    :param collections_size: Size of collections of all documents in index.
    :returns: Idf of passed token in index.
    """
    term_frequence = len(inverted_index.get(token, []))
    if term_frequence is 0:
        return 0
    return math.log((collection_size+1) / term_frequence)

def create_tf_for_token(article: str, doc_id: str, token: str):
    response = {
        'docID': doc_id,
        'tf': count_frequence(article, token)
    }
    return response

def summarize(matrix_of_tokens: List[str], docIds: List):
    """Create a inverted index with all tokens and yours docIds.
    :param matrix_of_tokens: matrix of article tokens lists.
    :param docIds: list of document ids of all articles.
    :returns: A inverted index with all tokens and yours docIds.
    """
    index = {}
    for i in range(len(matrix_of_tokens)):
        for token in set(matrix_of_tokens[i]):
            if token in index.keys():
                index[token].get('IDs').append(create_tf_for_token(matrix_of_tokens[i], docIds[i], token))
            else:
                index[token] = { 'IDs': [create_tf_for_token(matrix_of_tokens[i], docIds[i], token)] }
    
    for token in index.keys():
        index.get(token).update({ 'IDF': idf(index, token, matrix_of_tokens.size) })
    
    return index


def get_from_index_by_token(word: str, index: pandas.DataFrame):
    word = word.lower()
    if word in index.keys():
        return index.get(word).get('IDs')
    return []

def split_query(query: str):
    return list(map((lambda w: unidecode(w)), query.split()))


def search(index: pandas.DataFrame, query: str) -> Set[str]:
    """Search in inverted index using passed query.
    :param query: Query with two elements that will be searched in
                  inverted index and between them a conjunction or disjunction.
                  Example: "<word1> AND/OR <word2>"
    :returns: Return result of query execution on inverted index.
    """
    elements = split_query(query)
    operation = elements[QUERY_INDEX]
    
    result = []
    if operation == DISJUNCTION:
        result = list(get_from_index_by_token(elements[FIRST_WORD_INDEX], index))
        result.extend(list(get_from_index_by_token(elements[SECOND_WORD_INDEX], index)))
    else:
        result = set(get_from_index_by_token(elements[FIRST_WORD_INDEX], index)).intersection((
                    get_from_index_by_token(elements[SECOND_WORD_INDEX], index)
                 ))
    
    return set(result)

def conjunctive_search(query: str) -> int:
    """
    :param query: Query with n words that will be searched in
                  inverted index separated by space.
                  Example: "<word1> <word2> <word3> <word4>"
                 
    :returns: Return result of conjunction of the search between 
             all words on inverted index.
    """
    elements = split_query(query)
    
    index = {}
    for element in elements:
        index[len(inverted_index[element])] = element
    
    ordered_frequence = sorted(index.keys())
    
    # conjuntion between result of all elements.
    result = set(inverted_index[index[ordered_frequence[0]]])
    for i in range(1, len(ordered_frequence)):
        result = result.intersection(inverted_index[index[ordered_frequence[i]]])
        
    return result

def search_with_vectorial_model(query, inverted_index):
    elements = split_query(query)
    result = []
    for element in elements:
        result.extend(inverted_index[element]['IDs'])

    result = list(set(map((lambda x: x['docID']), result)))

    return sorted(result)[:5]


def contains(docId, lists):
    for l in lists:
        ids = list(map((lambda x: x.get('docID')), l))
        if not docId in ids:
            return False
    return True

def search_with_tf(query, inverted_index):
    elements = split_query(query)
    
    index = {}
    lists_of_ids = [inverted_index[element].get('IDs') for element in elements]
    for l in lists_of_ids:
        for obs in l:
            if obs.get('docID') in index.keys():
                index[obs.get('docID')] = (index[obs.get('docID')][0], index[obs.get('docID')][1] + obs.get('tf'))
            else:
                index[obs.get('docID')] = (obs.get('docID'), obs.get('tf'))
    
    index = index.values()
    index = sorted(index, key=lambda tup: tup[1], reverse=True)
    
    result = []
    i = 0
    while len(result) < 5 and i < len(index):
        if contains(index[i][0], lists_of_ids):
            result.append(index[i][0])
        i += 1
            
    return result


def search_with_tf_idf(query, inverted_index):
    elements = split_query(query)
    
    index = {}
    lists_of_ids = [inverted_index[element] for element in elements]
    for l in lists_of_ids:
        for obs in l.get('IDs'):
            if obs.get('docID') in index.keys():
                index[obs.get('docID')] = (index[obs.get('docID')][0], index[obs.get('docID')][1] + obs.get('tf') * l.get('IDF'))
            else:
                index[obs.get('docID')] = (obs.get('docID'), obs.get('tf') * l.get('IDF'))
    
    index = index.values()
    index = sorted(index, key=lambda tup: tup[1], reverse=True)
    
    result = []
    i = 0
    lists_of_ids = [l.get('IDs') for l in lists_of_ids]
    while len(result) < 5 and i < len(index):
        if contains(index[i][0], lists_of_ids):
            result.append(index[i][0])
        i += 1
            
    return result


def calc_bm25(tf):
    k = 5
    return ((k+1)*tf)/(tf+k)

def search_with_bm25(query, inverted_index):
    elements = split_query(query)
    
    index = {}
    lists_of_ids = [inverted_index[element] for element in elements]
     
    for l in lists_of_ids:
        for obs in l.get('IDs'):
            if obs.get('docID') in index.keys():
                index[obs.get('docID')] = (index[obs.get('docID')][0], index[obs.get('docID')][1] + calc_bm25(obs.get('tf')) * l.get('IDF'))
            else:
                index[obs.get('docID')] = (obs.get('docID'), ( calc_bm25(obs.get('tf')) * l.get('IDF')))
    
    index = index.values()
    index = sorted(index, key=lambda tup: tup[1], reverse=True)
    
    result = []
    i = 0
    lists_of_ids = [l.get('IDs') for l in lists_of_ids]
    while len(result) < 5 and i < len(index):
        if contains(index[i][0], lists_of_ids):
            result.append(index[i][0])
        i += 1
            
    return result

