import random
import csv
from more_itertools import unique_everseen

rel_dict = dict()
data_set = '../'


def tsv_writer(item_set, name):
    with open(name+'raw.txt', 'wb') as csvfile:
        datawriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE, delimiter='\t', skipinitialspace=True)
        for i in item_set:
            try:
                datawriter.writerow(i)
            except:
                pass


def tsv_writer_unique(name):
    with open(name+'raw.tsv', 'r') as f, open(name+'.txt', 'wb') as out_file:
        out_file.writelines(unique_everseen(f))


def corrupt(data):
    with open(data + '.txt') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            print row
            if row[1] not in rel_dict.keys():
                rel_dict[row[1]] = {'e1':[row[0]], 'e2':[row[2]]}
            else:
                rel_dict[row[1]]['e1'].append(row[0])
                rel_dict[row[1]]['e2'].append(row[2])


    suffle_rel = []
    for rel in rel_dict.keys():
        e1 = rel_dict[rel]['e1']
        e2 = rel_dict[rel]['e2']
        e3 = random.sample(e2, len(e2))
        for i, ent in enumerate(e1):
            if e3[i] != e2[i]:
                suffle_rel.append([ent, rel, e3[i], -1])
                suffle_rel.append([ent, rel, e2[i], 1])

    tsv_writer(suffle_rel, data)
    tsv_writer_unique(data)

# corrupt('spouseraw')
corrupt('test')