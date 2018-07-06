import csv


def tsv_writer(item_set, name):
    with open(name+'raw.txt', 'wb') as csvfile:
        datawriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE, delimiter='\t', skipinitialspace=True)
        for i in item_set:
            try:
                datawriter.writerow(i)
            except:
                pass

with open('data/Freebase/train.txt') as tsv:
    reader = csv.reader(tsv, delimiter='\t')
    spouse_count = 0
    child_count = 0
    spouse_list = []
    child_list = []
    other_list = []
    for row in reader:
    	if row[1] == 'spouse':
    		spouse_list.append(row)
    		spouse_count+=1
    	elif row[1] == 'children':
    		child_list.append(row)
    		child_count+=1
    	else:
    		other_list.append(row)

    	
    print spouse_count,child_count

# tsv_writer(spouse_list,'spouse')
# tsv_writer(child_list,'children')
tsv_writer(other_list,'other')