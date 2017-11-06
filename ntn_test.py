from load_data import getTestData, getDictionary, getWordIndices
from load_params import getProgramParameters
import numpy as np
import datetime
import cPickle as pickle
import sys


program_parameters = getProgramParameters()

data_set = program_parameters['data_set']
embedding_size = program_parameters['embedding_size']
slice_size = program_parameters['slice_size']
if data_set == 0:
    data_set = 'data/Wordnet/'
elif data_set == 1:
    data_set = 'data/Freebase/'
else:
    data_set = 'data/DBpedia/'

entity_dictionary, num_entities = getDictionary(data_set+'entities.txt')
relation_dictionary, num_relations = getDictionary(data_set+'relations.txt')
word_indices, num_words = getWordIndices(data_set+'wordIndices.p')




def paramsToStack(theta, decode_info):
    """ Initialize an empty stack """
    stack = []
    index = 0
    for i in range(len(decode_info)):
        """ Get the configuration of the 'i'th parameter to be put """
        decode_cell = decode_info[i]
        if isinstance(decode_cell, dict):
            """ If the cell is a dictionary, the 'i'th stack element is a dictionary """
            param_dict = {}
            for j in range(len(decode_cell)):
                """ Extract the parameter matrix from the unrolled vector """

                param_dict[j] = theta[index: index + np.prod(decode_cell[j])].reshape(decode_cell[j])
                index += np.prod(decode_cell[j])
            stack.append(param_dict)
        else:
            """ If not a dictionary, simply extract the parameter matrix """
            stack.append(theta[index: index + np.prod(decode_cell)].reshape(decode_cell))
            index += np.prod(decode_cell)
    return stack


def getPredictions(test_data):
    """ Get stack of network parameters """
    theta = np.loadtxt(data_set+'params/theta.txt')
    best_thresholds = np.loadtxt(data_set+'params/thresholds.txt')

    with open(data_set+'params/decode_info.p', 'rb') as fp:
        decode_info = pickle.load(fp)

    W, V, b, U, word_vectors = paramsToStack(theta, decode_info)

    """ Initialize entity vectors as matrix of zeros """
    entity_vectors = np.zeros((embedding_size, num_entities))

    """ Assign entity vectors to be the mean of word vectors involved """
    for entity in range(num_entities):
        entity_vectors[:, entity] = np.mean(word_vectors[:, word_indices[entity]], axis=1)

    """ Initialize predictions as an empty array """
    predictions = np.empty((test_data.shape[0], 1))

    for i in range(test_data.shape[0]):
        """ Extract required information from 'test_data' """
        rel = test_data[i, 1]
        entity_vector_e1 = entity_vectors[:, test_data[i, 0]].reshape(embedding_size, 1)
        entity_vector_e2 = entity_vectors[:, test_data[i, 2]].reshape(embedding_size, 1)

        """ Stack the entity vectors one over the other """
        entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))
        test_score = 0

        """ Calculate the prediction score for the 'i'th example """
        for k in range(slice_size):
            test_score += U[rel][k, 0] * \
                          (np.dot(entity_stack.T, np.dot(W[rel][:, :, k], entity_stack)) +
                           np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])

        """ Give predictions based on previously calculate thresholds """
        if (test_score <= best_thresholds[rel]):
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1

    return predictions

for rel in relation_dictionary.keys():
    test_data, test_labels = getTestData(data_set +'test/'+ rel +'.txt', entity_dictionary, relation_dictionary)
    predictions = getPredictions(test_data)

    """ Print accuracy of the obtained predictions """
    print rel
    print "Accuracy:", np.mean((predictions == test_labels))
    # accuracy = np.mean((predictions == test_labels))
    # f = open('test_accuracy.txt', 'a')
    # f.write(str(datetime.datetime.now()) + '\t' + str(accuracy) + '\n')
    # f.close()

