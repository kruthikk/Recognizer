import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    p is the number of parameters, and N is the number of data points - features.
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        """
        # TODO implement model selection based on BIC scores
        #https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
        #Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities    
        # n = no of states, d = no of features
        #Transition probs = n*(n-1) = n*n-n
        #Starting probabilities =  n - 1
        # no of means variances = 2* n*d
        # =  n*n + 2*n*d-1
        """
        bestscore = float("inf")
        bestmodel = self.base_model(self.n_constant)

        
        nfeatures = sum(self.lengths)
        for nstates in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(nstates)
                logL = model.score(self.X, self.lengths)        
                
                #no of parametrs = n*n + 2*n*d-1
                nparams = nstates * nstates + 2 * nstates * nfeatures - 1
                #bic = -2 * logL + p * logN
                bic = -2 * logL + nparams * np.log(nfeatures)
                
                if (bic < bestscore and model is not None):
                    bestscore, bestmodel = bic, model

            except Exception as e:
                pass

        return bestmodel

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        """
        # TODO implement model selection based on DIC scores
        
        From reference - Discriminitaing factor = The ratio of the evidence and
        the anti-evidence is thus a measure of the model capacity to
        discriminate data from the two competing classe
        Di =  P (Xi | Til) /   P (Xj | Til )
        DFC = logarithm of Di
        Discriminative Information Criterion (DIC) =  approximation of the DFC --> BIC approximation
        If all datasets are of same size
        DIC =  likelihood of the data - the average of anti-likelihood 
                The anti-likelihood of the data Xj against model M is a likelihood-like quantity in
                which the data and the model belong to competing categories.
        from forums..
        log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        Hint log(P(X(i)) is the result of the score function, which is returned by model.score(X, lenghts)
        
        log(P(given word)) - Avg(log(P(all other words))) 
        
        """
        bestscore = float("-inf")
        bestmodel = self.base_model(self.n_constant)
        
        for nstates in range(self.min_n_components, self.max_n_components + 1):
            try:
                nstatewordsmodel = {}
                nstatewordslogL = {}

                for word in self.words.keys():
                    try:
                        X,lengths = self.hwords[word]
                        wordmodel = GaussianHMM(n_components=nstates, covariance_type="diag", n_iter=1000,
                                    random_state=inst.random_state, verbose=False).fit(X, lengths)
                        wordlogL = wordmodel.score(X,lengths)
                        nstatewordsmodel[word] = wordmodel
                        nstatewordslogL[word] = wordlogL
                    except Exception as e:
                        continue
               
                if (self.this_word not in nstatewordslogL):
                    continue

                givenwordlikelihood = nstatewordslogL[self.this_word]
                avg_anti_likelihood = np.average([nstatewordslogL[word] for word in nstatewordslogL.keys() if word != self.this_word] )

                dic = givenwordlikelihood - avg_anti_likelihood

                if (dic > bestscore and nstatewordsmodel[self.this_word] is not None):
                    bestscore, bestmodel = dic, nstatewordsmodel[self.this_word]
                    
            except Exception as e:
                continue

        
        return bestmodel

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        bestscore = float("-inf")
        bestmodel = self.base_model(self.n_constant)
        
        split_method = KFold()
        for nstates in range(self.min_n_components, self.max_n_components + 1):
            try:
                logLScores = []
                model, logL, avgscore = None, None, float("-inf")

                if (len(self.sequences) < split_method.n_splits):
                    model = self.base_model(nstates) 
                    logL = model.score(self.X, self.lengths)
                    logLScores.append(logL)
                else:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        Xtest, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        try:
                            model = None
                            model = GaussianHMM(n_components= nstates, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                            logL = model.score(Xtest, lengths_test)
                            logLScores.append(logL)
                        except Exception as e:
                            continue

                if (len(logLScores) > 0 ):
                    avgscore = np.average(logLScores)
                if (avgscore > bestscore and model is not None):
                    bestscore, bestmodel = avgscore, model

            except Exception as e:
                pass

        return bestmodel
            



