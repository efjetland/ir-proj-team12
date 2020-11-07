import elasticsearch
import requests
import json
import re

from collections import defaultdict
from elasticsearch import Elasticsearch



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
    return [x[0] for x in scores]


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
    return [x[0] for x in scores]

def score(queries, es, index_name, scorer, k):
    query_scores = []
    for query in queries:
        scores = scorer(es, index_name, query['question'], k=k)
        query_scores.append({'id': query['id'], 'category': 'resource', 'type': scores})
    return query_scores
        


def main():
    es = Elasticsearch()
    TRAIN_QUERY, _ = fetch_queries()
    INDEX_NAME_ENTITY = 'nlp_entity'
    INDEX_NAME_TYPE = 'nlp_type'

    ec_100 = score(TRAIN_QUERY, es, INDEX_NAME_ENTITY, entity_centric_scorer, 100)
    tc_100 = score(TRAIN_QUERY, es, INDEX_NAME_TYPE, type_centric_scorer, 100)
    with open('ec_100.json', 'w') as output:
        json.dump(ec_100, output)
    with open('tc_100.json', 'w') as output:
        json.dump(tc_100, output)
    with open('gold.json', 'w') as output:
        json.dump(TRAIN_QUERY, output)


if __name__ == "__main__":
    main()



