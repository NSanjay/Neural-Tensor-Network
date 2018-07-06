stats = {}
stats_set = {}
all_persons = []


with open("data/Freebase/train.txt") as f:
	content = f.readlines()
	for line in content:
		persons = line.split('\t')
		predicate = persons[1]
		persons.pop(1)
		persons = [per.rstrip() for per in persons]
		persons = [per.replace('-', '_') for per in persons]
		if predicate not in stats.keys():
			stats[predicate] = [persons]
			stats_set[predicate] = [persons[1]]
		else:
			stats[predicate].append(persons)
			if persons[1] not in stats_set[predicate]:
				stats_set[predicate].append(persons[1])
		all_persons.extend(persons)

total_data = 0

for k,v in stats.iteritems():
	print k, len(stats[k])
	total_data = total_data + len(stats[k])
print total_data

all_persons_unique = list(set(all_persons))
print "unique entities===>"
print len(all_persons_unique)

total_data_u = 0
print "individual entities==>"
for k,v in stats_set.iteritems():
	print k, len(stats_set[k])
	total_data_u = total_data_u + len(stats_set[k])
print total_data_u