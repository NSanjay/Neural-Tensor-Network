import numpy as np
import pickle
import sys
###########################################################################################
""" Read and construct training data, using entity and relation dictionaries """


def getTrainingData(file_name, entity_dictionary, relation_dictionary):

    """ Read and split data linewise """

    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()

    """ Initialize training data as an empty matrix """

    num_examples = len(data)
    training_data = np.empty((num_examples, 3), dtype=int)

    index = 0

    for line in data:
        """ Obtain relation example text by splitting line """
        entity1, relation, entity2 = line.split()

        """ Assign indices to the obtained entities and relation """
        training_data[index, 0] = entity_dictionary[entity1]
        training_data[index, 1] = relation_dictionary[relation]
        training_data[index, 2] = entity_dictionary[entity2]
        index += 1

    return training_data, num_examples


""" Read and construct test data, using entity and relation dictionaries """


def getTestData(file_name, entity_dictionary, relation_dictionary):

    """ Read and split data linewise """

    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()

    """ Initialize test data and labels as empty matrices """

    num_entries = len(data)
    test_data = np.empty((num_entries, 3), dtype=int)
    labels = np.empty((num_entries, 1))

    index = 0

    for line in data:
        """ Obtain relation example text by splitting line """

        entity1, relation, entity2, label = line.split()

        """ Assign indices to the obtained entities and relation """

        test_data[index, 0] = entity_dictionary[entity1]
        test_data[index, 1] = relation_dictionary[relation]
        test_data[index, 2] = entity_dictionary[entity2]

        """ Label value for the example """

        if label == '1':
            labels[index, 0] = 1
        else:
            labels[index, 0] = -1

        index += 1

    return test_data, labels

""" Create a numerical mapping of entity/relation data elements """


def getDictionary(file_name):

    """ Read and split data linewise """

    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()

    """ Initialize dictionary to store the mapping """

    dictionary = {}
    index = 0

    for entity in data:

        """ Assign unique index to every entity """

        dictionary[entity] = index
        index += 1

    """ Number of entries in the data file """

    num_entries = index

    return dictionary, num_entries

""" Get indices of words in the entities """


def getWordIndices(file_name):

    """ Load the pickled data file """

    word_dictionary = pickle.load(open(file_name, 'rb'))

    """ Extract the number of words and the word indices from the dictionary """

    num_words = word_dictionary['num_words']
    word_indices = word_dictionary['word_indices']
    # print num_words, len(word_indices)
    return word_indices, num_words
