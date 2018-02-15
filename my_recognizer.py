import warnings
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
    """
    https://discussions.udacity.com/t/recognizer-implementation/234793/21
    We are trying to predict using the models and the test_set.
    The method receives a HMM model for each word. For example, one model for the word FISH, one model for the word CHOCOLATE, so on.
    Then, we got a data set (test_set) that are X and lengths values. For each one, we want to determine which word is the most likely. 
    We are going to do that calculating the score for each data in all the models, and selecting the word corresponding 
    to the highest score model.
    guess is going to be, for each item in the test_set, the word with the highest probability model.
    and probabilities is going to have for each item in the test_set, a dictionary with the possible word and the score of the model.
    # TODO implement the recognizer
    # return probabilities, guesses
    """

    """
          for each test word
            for word, model in models.items():
>            calculate the scores for each model(word) and update the 'probabilities' list.
>            determine the maximum score for each model(word).
>            Append the corresponding word (the tested word is deemed to be the word for which with the model was trained) to the list 'guesses'.
    """
    

    for wordid in range(0, len(test_set.get_all_Xlengths())):
      bestscore = float("-inf")
      bestword = None
      testwordprobabilities = {}
      try:
        for word, model in models.items():
          X, lengths = test_set.get_item_Xlengths(wordid)
          score = float("-inf")
          try:
            score = model.score(X, lengths)
          except Exception as e:
            pass
          testwordprobabilities[word] = score
          if (score > bestscore):
            bestscore , bestword = score, word
      except Exception as e:
        pass
      
      guesses.append(bestword)
      probabilities.append(testwordprobabilities)

    return probabilities, guesses
