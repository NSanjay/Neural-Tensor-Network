
def getProgramParameters():

    """ Initialize dictionary of program parameters """

    program_parameters = {}

    print(""" Set program parameters """)

    program_parameters['embedding_size'] = 100    # size of a single word vector
    program_parameters['slice_size'] = 3      # number of slices in tensor
    program_parameters['num_iterations'] = 50    # number of optimization iterations
    program_parameters['batch_size'] = 20000  # training batch size
    program_parameters['corrupt_size'] = 10     # corruption size
    program_parameters['activation_function'] = 0      # 0 - tanh, 1 - sigmoid
    program_parameters['lamda'] = 0.0001 # regulariization parameter
    program_parameters['batch_iterations'] = 5      # optimization iterations for each batch
    program_parameters['data_set'] = 0      # 0-Wordnet 1-Freebase

    return program_parameters

###########################################################################################