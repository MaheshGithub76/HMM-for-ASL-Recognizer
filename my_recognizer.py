import warnings
import numpy as np
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    # raise NotImplementedError

    opt_guess = None

    for indx in range(0, len(test_set.get_all_Xlengths())):
        opt_logL = float('-inf')
        X, lengths = test_set.get_all_Xlengths()[indx]
        prob_dict = {}
        for key, mdl in models.items():
            try:
                logL = mdl.score(X, lengths)
                prob_dict[key] = logL
            except:
                logL = float('-inf')
                prob_dict[key] = logL

            if logL > opt_logL:
                opt_logL = logL
                opt_guess = key

        probabilities.append(prob_dict)
        guesses.append(opt_guess)

    return probabilities, guesses


