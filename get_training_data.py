import sparql
import csv
from more_itertools import unique_everseen

sparql_dbpedia = 'http://dbpedia.org/sparql'

def get_all_person():
    q_ts = 'PREFIX foaf:  <http://xmlns.com/foaf/0.1/> PREFIX dbo:  <http://dbpedia.org/ontology/> \
    PREFIX dbp: <http://dbpedia.org/property/> select ?url1  ?url2 ?url3 ?url4 ?url5 ?url6 ?url7 ?url8 ?url9 ?url10 ?url11\
     ?url12 ?url13 ?url14 where  { ?url1  dbo:birthPlace ?url6 .  ?url1  foaf:gender ?url2 . optional { ?url1 \
      dbp:nationality ?url3} . optional { ?url1  dbo:deathPlace ?url5 }. \
      optional {   ?url1  dbo:profession ?url4 }. optional { ?url1  dbo:residence ?url7} . optional { ?url1  dbo:almaMater \
      ?url8 }. optional { ?url1  dbo:deathCause ?url9 }. optional { ?url1  dbo:religion ?url10 } .  optional { ?url1  dbo:parent ?url11} \
      . optional { ?url1  dbo:child ?url12} . optional { ?url1  dbo:ethnicity ?url13} .  optional { ?url1 dbo:spouse ?url14} .  }'
    print q_ts
    result = sparql.query(sparql_dbpedia, q_ts)
    training_set = [sparql.unpack_row(row_result) for row_result in result]
    return training_set


def tsv_writer(item_set):
    with open('pre_data/raw_triples.tsv', 'wb') as csvfile:
        datawriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE, delimiter='\t', skipinitialspace=True)
        for i in item_set:
            try:
                datawriter.writerow(i)
            except:
                pass


def tsv_writer_unique():
    with open('pre_data/raw_triples.tsv', 'r') as f, open('pre_data/unique_raw_triples.tsv', 'wb') as out_file:
        out_file.writelines(unique_everseen(f))

train_data = get_all_person()
print len(train_data)

entity_set = []
filtered_evidence = []
rule_predicates = ["gender", "nationality", "profession", "place_of_death", "place_of_birth", "location", "institution",\
                   "cause_of_death", "religion", "parents", "children", "ethnicity", "spouse"]

related_pred = ["parent", "child", "spouse"]
related_person = []

for i, entities in enumerate(train_data):
    url = entities[0].split('/')[-1]
    entity_set.append(url)
    entities.pop(0)
    for j, entity in enumerate(entities):
        if entity:
            if isinstance(entity, basestring):
                if '/' in entity:
                    entity = entity.split('/')[-1]
                filtered_evidence.append([url, rule_predicates[j], entity])
                if rule_predicates[j] in related_pred:
                    if entity not in related_person:
                        related_person.append(entity)
                entity_set.append(entity)


tsv_writer(filtered_evidence)
tsv_writer_unique()

with open('pre_data/related_persons.txt', 'wb') as personwriter:
    for per in related_person:
        try:
            personwriter.write(per + '\n')
        except:
            print per