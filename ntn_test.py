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
w_params = program_parameters['w_param']

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
    print data_set
    theta = np.loadtxt(data_set + 'parameters/theta.txt')

    with open(data_set + 'parameters/decode_info.p', 'rb') as fp:
        decode_info = pickle.load(fp)
    best_thresholds = np.loadtxt(data_set+'parameters/thresholds.txt')
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
            if w_params == 1:
                test_score += U[rel][k, 0] * \
                              (np.dot(entity_stack.T, np.dot(W[rel][:, :, k], entity_stack)) +
                               np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])
            else:
                test_score += U[rel][k, 0] * \
                              (np.dot(entity_vector_e1.T, np.dot(W[rel][:, :, k], entity_vector_e2)) +
                               np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])
        """ Give predictions based on previously calculate thresholds """
        if (test_score <= best_thresholds[rel]):
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1
        # predictions[i, 1] = test_score
        # predictions[i, 2] = best_thresholds[rel]
    return predictions


def computeBestThresholds(dev_data, dev_labels, data_set, theta, decode_info):
    W, V, b, U, word_vectors = paramsToStack(theta, decode_info)
    entity_vectors = np.zeros((embedding_size, num_entities))
    """ Assign entity vectors to be the mean of word vectors involved """
    for entity in range(num_entities):
        entity_vectors[:, entity] = np.mean(word_vectors[:, word_indices[entity]], axis=1)

    dev_scores = np.zeros(dev_labels.shape)
    for i in range(dev_data.shape[0]):
        """ Extract required information from 'dev_data' """
        rel = dev_data[i, 1]
        entity_vector_e1 = entity_vectors[:, dev_data[i, 0]].reshape(embedding_size, 1)
        entity_vector_e2 = entity_vectors[:, dev_data[i, 2]].reshape(embedding_size, 1)
        """ Stack the entity vectors one over the other """
        entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))
        """ Calculate the prediction score for the 'i'th example """
        for k in range(slice_size):
            if w_params == 0:
                dev_scores[i, 0] += U[rel][k, 0] * \
                                    (np.dot(entity_vector_e1.T, np.dot(W[rel][:, :, k], entity_vector_e2)) +
                                     np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])
            else:
                dev_scores[i, 0] += U[rel][k, 0] * \
                                    (np.dot(entity_stack.T, np.dot(W[rel][:, :, k], entity_stack)) +
                                     np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])

    """ Minimum and maximum of the prediction scores """
    score_min = np.min(dev_scores)
    score_max = np.max(dev_scores)
    """ Initialize thresholds and accuracies """
    best_thresholds = np.empty((num_relations, 1))
    best_accuracies = np.empty((num_relations, 1))
    for i in range(num_relations):
        best_thresholds[i, :] = score_min
        best_accuracies[i, :] = -1

    score_temp = score_min
    interval = 0.01

    """ Check for the best accuracy at intervals between 'score_min' and 'score_max' """
    while(score_temp <= score_max):
        for i in range(num_relations):
            """ Check accuracy for 'i'th relation at 'score_temp' """
            rel_i_list = (dev_data[:, 1] == i)
            predictions = (dev_scores[rel_i_list, 0] <= score_temp) * 2 - 1
            temp_accuracy = np.mean((predictions == dev_labels[rel_i_list, 0]))
            """ If the accuracy is better, update the threshold and accuracy values """
            if(temp_accuracy > best_accuracies[i, 0]):
                best_accuracies[i, 0] = temp_accuracy
                best_thresholds[i, 0] = score_temp
        score_temp += interval
    """ Store the threshold values to be used later """
    # print "Best Threshold: " + str(best_thresholds)
    best_thresholds = best_thresholds
    np.savetxt(data_set+'thresholds.txt', best_thresholds)




# dev_data, dev_labels = getTestData(data_set+'dev.txt', entity_dictionary, relation_dictionary)
#
# computeBestThresholds(dev_data, dev_labels, data_set, theta, decode_info)

# for rel in relation_dictionary.keys():
#     test_data, test_labels = getTestData(data_set + 'test/' + rel + '.txt', entity_dictionary, relation_dictionary)
#     predictions = getPredictions(test_data, theta, decode_info)
#     """ Print accuracy of the obtained predictions """
#     print str(rel)+" Accuracy:", np.mean((predictions == test_labels))
#     print type(predictions)
#     print type(test_labels)
#     np.savetxt(str(rel)+"predictions_fc.csv", predictions, delimiter=",")
#     np.savetxt(str(rel)+"test_fc_labels.csv", test_labels, delimiter=",")
#    # accuracy = np.mean((predictions == test_labels))
#     # f = open('test_accuracy.txt', 'a')
#     # f.write(str(datetime.datetime.now()) + '\t' + str(accuracy) + '\n')
#     # f.close()

test_data, test_labels = getTestData(data_set + 'test.txt', entity_dictionary, relation_dictionary)


predictions = getPredictions(test_data)
print predictions.shape
np.savetxt(data_set +"parameters/predictions_all.csv", predictions, delimiter=",")

accuracy = np.mean((predictions == test_labels))
print accuracy
f = open('accuracy.txt', 'a')
f.write(str(datetime.datetime.now()) + '\t' +'fc' + '\t' + str(accuracy) + '\n')
f.close()

