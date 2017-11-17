import pandas as pd
import numpy as np
import csv
import sys


data_set = '../'
df = pd.read_csv(data_set+'all_person.txt', sep='/t')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]

del train['split']
del test['split']

train.to_csv(data_set+'train_1.txt', sep='\t', header=False, index=False)
test.to_csv(data_set+'test.txt', sep='\t', header=False, index=False)
#remove the quotes
sys.exit(0)
df = pd.read_csv(data_set+'train.txt', sep='/t')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
dev = df[~msk]

del train['split']
del dev['split']

train.to_csv(data_set+'train.txt', sep='\t', header=False, index=False)

dev.to_csv(data_set+'dev.txt', sep='\t', header=False, index=False)

def relation_counter(data):
    relation_set = []
    with open(data_set+data+'.txt') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            relation_set.append(row[1])
        relation_set_full = relation_set

    relation_count_dict = {i:relation_set_full.count(i) for i in set(relation_set_full)}
    print relation_count_dict

rel_data = ['train_comb', 'test_comb', 'dev_comb']

for data in rel_data:
    relation_counter(data)
