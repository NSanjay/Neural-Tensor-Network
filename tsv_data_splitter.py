import pandas as pd
import csv

data_set = 'data/DBpedia_2/'

# with open(data_set+'unique_raw_triples.tsv', 'wb') as csvfile:
#     all_data = csv.reader(csvfile, quoting=csv.QUOTE_NONE, delimiter='\t', skipinitialspace=True)
#
# print "all_data",all_data
all_data = pd.read_csv(data_set+'unique_raw_triples.tsv', sep='\t')

train_dev = all_data.sample(frac=0.8,random_state=200)
test = all_data.drop(train_dev.index)
train = train_dev.sample(frac=0.9,random_state=200)
dev = train_dev.drop(train.index)

dev.to_csv(data_set+'dev.txt', sep='\t', header=False, index=False)
train.to_csv(data_set+'train.txt', sep='\t', header=False, index=False)
test.to_csv(data_set+'test.txt', sep='\t', header=False, index=False)