import elasticsearch
import requests
import json
import re
import numpy as np
import math
from nltk.corpus import wordnet,stopwords
from nltk import word_tokenize, pos_tag
import gensim.downloader as api
from gensim.models import Word2Vec

from evaluation import load_type_hierarchy
from collections import defaultdict
from elasticsearch import Elasticsearch
from sklearn.ensemble import RandomForestRegressor



def fetch_queries():
    url = 'https://raw.githubusercontent.com/smart-task/smart-dataset/master/datasets/DBpedia/'
    file = 'smarttask_dbpedia_train.json'
    url = url + file
    queries = requests.get(url).json()
    queries = process_queries(queries)

    TRAIN_SIZE = int(len(queries) * 0.8)

    TRAIN_QUERY = [q for q in queries[:TRAIN_SIZE] if q['category'] == 'resource']
    TEST_QUERY = queries[TRAIN_SIZE:]

    print(len(TRAIN_QUERY))
    print(len(TEST_QUERY))
    return TRAIN_QUERY, TEST_QUERY

def process_queries(queries):
    processed = []
    pattern = re.compile(r'[\W_]+')
    for query in queries:
        if query['question'] == None:
            continue
        query['question'] = pattern.sub(' ', query['question'])
        processed.append(query)
    return processed

def analyze_query(es, index_name, query):
    tokens = es.indices.analyze(index=index_name, body={'text': query})['tokens']
    query_terms = []
    for t in sorted(tokens, key=lambda x: x['position']):
        query_terms.append(t['token'])
    return query_terms

def entity_centric_scorer(es, index_name, query, field='description', k=100):
    es_query = {
        "query":{
            "query_string":{
                "query": query,
                "default_field": field
            }
        }
    }

    matches = es.search(index=index_name, body=es_query, _source=True, size=k)['hits']['hits']

    type_count = defaultdict(int)
    for match in matches:
        for doc_type in match['_source']['types']:
            type_count[doc_type] += 1

    type_weight = {}
    for t, c in type_count.items():
        type_weight[t] = 1/c

    type_scores = defaultdict(int)
    for match in matches:
        doc_score = match['_score']
        for doc_type in match['_source']['types']:
            type_scores[doc_type] += doc_score * type_weight[doc_type]

    scores = sorted([(t, s) for t, s in type_scores.items()])
    scores = sorted(scores,key=lambda tup: tup[1], reverse=True)
    return scores

def type_centric_scorer(es, index_name, query, field='description', k=100):
    type_scores = defaultdict(int)
    q_terms = analyze_query(es, index_name, query)
    for term in q_terms:
        es_query = {
            "query":{
                "query_string":{
                    "query": term,
                    "default_field": field
                }
            }
        }

        matches = es.search(index=index_name, body=es_query, _source=False, size=k)['hits']['hits']
        for match in matches:
            type_scores[match['_id']] += match['_score']

    scores = sorted([(t, s) for t, s in type_scores.items()])
    scores = sorted(scores,key=lambda tup: tup[1], reverse=True)
    return scores

def score(queries, es, index_name, scorer, field='description', k=100):
    query_scores = {}
    for query in queries:
        scores = scorer(es, index_name, query['question'], field=field, k=k)
        scores = minmax(scores, 10)
        query_scores[query['id']] = {'category': 'resource', 'type': scores}
    return query_scores

def sort_scores(scores):
    results = []
    for q in scores.keys():
        t_scores = [(k, v) for k,v in scores[q]['type'].items()]
        t_scores = sorted(t_scores)
        t_scores = [t[0] for t in sorted(t_scores, key=lambda tup: tup[1], reverse=True)]
        results.append({'id': q, 'category': scores[q]['category'], 'type': t_scores})
    return results
        
def minmax(scores, n):
    min = 1000000
    max = 0
    for _, s in scores:
        if s < min:
            min = s
        if s > max:
            max = s 
    minmaxed = {}
    if max == 0 or max == min:
        for t, s in scores[:n]:
            minmaxed[t] = 0
        return minmaxed
    for t, s in scores[:n]:
        minmaxed[t] = (s - min)/(max-min)
    return minmaxed

class PointWiseLTRModel(object):
    def __init__(self, regressor):
        """
        Args:
            classifier: an instance of scikit-learn regressor
        """
        self.regressor = regressor

    def _train(self, X, y):
        """Trains and LTR model.
        
        Args:
            X: features of training instances
            y: relevance assessments of training instances
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(self, ft, doc_ids):
        """Predicts relevance labels and rank documents for a given query.
        
        Args:
            ft: a list of features for query-doc pairs
            doc_ids: a list of document ids
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results

def idf_type_feature (t_terms,queries,t_features):
    idf=[]
    
    if t_terms is not None:
        num_doc = len(queries)
        length = sum(len(i) for i in t_terms)
        
        for t in t_terms:
            df = 0
            for key in queries:
                if t in queries[key]: 
                    df=df+1 
            
            if df == 0: idf.append(0)
            else: idf.append(math.log(num_doc)/df)

        if len(idf)>0:
            sum_idf = sum(idf)
            avg_idf = sum(idf)/len(t_terms)
        else:
            sum_idf = 0
            avg_idf = 0
        
    else:
        sum_idf = 0
        avg_idf = 0
        length = 0
    
    t_features.append(length)
    t_features.append(sum_idf)
    t_features.append(avg_idf)
    return t_features

def jterms (t_terms,q_terms,t_features):
    if t_terms is not None and q_terms is not None and len(q_terms) > 0:
        t_terms = set(t_terms)
        q_terms = set([item for sublist in q_terms for item in sublist])
        jsim = float(len(t_terms.intersection(q_terms))) / (len(t_terms) + len(q_terms) - len(t_terms.intersection(q_terms)))
    else:
        jsim = 0
    t_features.append(jsim)
    return t_features

def similarity(t_terms, q_term, model, t_features):
    similarity = []
    #q = list(chain.from_iterable(term))
    for q in q_term:
        if q is not None:
            for t in t_terms:
                try:
                    similarity.append(model.wv.similarity(t,q))
                except:
                    similarity.append(0)
    
    if len(similarity) >0:
        t_features.append(max(similarity))
        t_features.append(sum(similarity))
        t_features.append(sum(similarity)/len(similarity))
    else:
        t_features.append(0)
        t_features.append(0)
        t_features.append(0)
        
    return t_features

def gettypelabelterm(dbotype):
    t_term = [t.lower() for t in re.findall('.[^A-Z]*',dbotype)[1:]]
    
    return t_term

def getqueryterm(queries):
    q_terms = {}
    stop_words = set(stopwords.words('english'))
    
    for query in queries:
        if query['question']:
            question = query['question'].lower()
            question = re.sub(r'[-()\"#/@;:<>{}`+=~|.!?,]', '', question).split(' ')
            question = [q for q in question if not q in stop_words]
            question =[q for q in question if len(q)>1]           
            q_terms[query['id']]=question
    return q_terms

def typelabel_features(t_term, q_term, queries, w2vm):
    '''input: dbotype query term, and query collection'''
    typefeature = {'sum_idf','avg_idf','length','jsim','avg_sim'}
    t_features=[]
    
    #for q_term in q_terms:
    t_features = idf_type_feature(t_term, queries, t_features)
    t_features = jterms(t_term, q_term, t_features)
    t_features = similarity(t_term, q_term, w2vm, t_features)
    return t_features

def get_features(t_term,q_term, queries, w2vmodel,baselinefeatures):
    '''combine both type label and baseline features '''
    typefeatures = typelabel_features(t_term, q_term, queries, w2vmodel) 
    
    return typefeatures + baselinefeatures

def get_feature_vectors(queries, ec_scorers, tc_scorers, qt_scorers, hierarchy):
    fvecs = []
    gts = []
    qids = []
    tids = []
    w2vmodel = api.load('word2vec-google-news-300')
    qs = getqueryterm(queries)

    for query in queries:
        qid = query['id']
        types = set()
        for i in range(len(ec_scorers)):
            ec_types = ec_scorers[i][qid]['type']
            tc_types = tc_scorers[i][qid]['type']
            qt_types = qt_scorers[i][qid]['type']
            types.update(ec_types.keys())
            types.update(tc_types.keys())
            types.update(qt_types.keys())

        for t in types:
            fv = []
            for i in range(len(ec_scorers)):
                ec_types = ec_scorers[i][qid]['type']
                tc_types = tc_scorers[i][qid]['type']
                qt_types = qt_scorers[i][qid]['type']
                if t in ec_types:
                    fv.append(ec_types[t])
                else:
                    fv.append(0)
                if t in tc_types:
                    fv.append(tc_types[t])
                else:
                    fv.append(0)
                if t in qt_types:
                    fv.append(qt_types[t])
                else:
                    fv.append(0)
            if t in hierarchy:
                fv.append(hierarchy[t]['depth'])
                if 'type' in query and t in query['type']:
                    gts.append(hierarchy[t]['depth'])
                else:
                    gts.append(0)
            else:
                fv.append(0)
                gts.append(0)
            t_term = gettypelabelterm(t)
            q_term = qs.get(query['id'],None)
            fvecs.append(get_features(t_term, q_term, qs, w2vmodel, fv))
            qids.append(qid)
            tids.append(t)

    return fvecs, gts, qids, tids


def main():
    es = Elasticsearch()
    TRAIN_QUERY = process_queries(json.load(open('data/train_query.json', 'r')))
    TEST_QUERY = process_queries(json.load(open('data/test_query.json', 'r')))
    INDEX_NAME_ENTITY = 'nlp_entity'
    INDEX_NAME_TYPE = 'nlp_type'
    types, _ = load_type_hierarchy('dbpedia_types.tsv')

    # Training features
    print("Scoring training queries")
    ec_100 = score(TRAIN_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=100)
    tc_100 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=100)
    qt_100 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 100)

    ec_50 = score(TRAIN_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=50)
    tc_50 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=50)
    qt_50 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 50)

    ec_20 = score(TRAIN_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=20)
    tc_20 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=20)
    qt_20 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 20)

    ec_10 = score(TRAIN_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=10)
    tc_10 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=10)
    qt_10 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 10)

    ec_5 = score(TRAIN_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=5)
    tc_5 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=5)
    qt_5 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 5)

    ec_train = [ec_100, ec_50, ec_20, ec_10, ec_5]
    tc_train = [tc_100, tc_50, tc_20, tc_10, tc_5]
    qt_train = [qt_100, qt_50, qt_20, qt_10, qt_5]

    # Test features
    print("Scoring test queries")
    ec_100_test = score(TEST_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=100)
    tc_100_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=100)
    qt_100_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 100)

    ec_50_test = score(TEST_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=50)
    tc_50_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=50)
    qt_50_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 50)

    ec_20_test = score(TEST_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=20)
    tc_20_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=20)
    qt_20_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 20)

    ec_10_test = score(TEST_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=10)
    tc_10_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=10)
    qt_10_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 10)

    ec_5_test = score(TEST_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, k=5)
    tc_5_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, k=5)
    qt_5_test = score(TEST_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer,'question', 5)

    with open('ec_test.json', 'w') as output:
        json.dump(sort_scores(ec_100_test), output)
    with open('tc_test.json', 'w') as output:
        json.dump(sort_scores(tc_100_test), output)
    with open('qt_test.json', 'w') as output:
        json.dump(sort_scores(qt_100_test), output)

    ec_test = [ec_100_test, ec_50_test, ec_20_test, ec_10_test, ec_5_test]
    tc_test = [tc_100_test, tc_50_test, tc_20_test, tc_10_test, tc_5_test]
    qt_test = [qt_100_test, qt_50_test, qt_20_test, qt_10_test, qt_5_test]

    
    X_train, y_train, _, _ = get_feature_vectors(TRAIN_QUERY, ec_train, tc_train, qt_train, types)
    clf = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=10)
    ltr = PointWiseLTRModel(clf)
    print('Training')
    for i in range(10):
        print('Data: {}\nLabel: {}'.format(X_train[i], y_train[i]))
    ltr._train(X_train, y_train)
    print('Finished training')

    X_test, _, qids_test, tids_test = get_feature_vectors(TEST_QUERY, ec_test, tc_test, qt_test, types)
    scores = []
    cur_qid = qids_test[0]
    X = []
    tids = []
    print('Ranking')
    for i, qid in enumerate(qids_test):
        if qid != cur_qid:
            r = ltr.rank(X, tids)
            r = sorted(r,key=lambda tup: tup[1], reverse=True)
            scores.append({'id': cur_qid, 'category': 'resource', 'type': [t[0] for t in r]})
            cur_qid = qid
            X = []
            tids = []
        X.append(X_test[i])
        tids.append(tids_test[i])
    r = ltr.rank(X, tids)
    r = sorted(r,key=lambda tup: tup[1], reverse=True)
    scores.append({'id': cur_qid, 'category': 'resource', 'type': [t[0] for t in r]})
    with open('system_ranking.json', 'w') as output:
        json.dump(scores, output)
    with open('gold.json', 'w') as output:
        json.dump(TEST_QUERY, output)




if __name__ == "__main__":
    main()



