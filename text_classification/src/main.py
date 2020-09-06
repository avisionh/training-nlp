import numpy as np
import logging
import gensim
import nltk

from gensim.models.doc2vec import TaggedDocument

def word_averaging(wv, words):
    """ Averages the words being passed in from word2vec, wv, model

    :param wv:
    :param words:

    :return:
    """

    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            #FIXME: remove these examples in preprocessing
            return np.zeros(wv.vector_size,)

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean


def word_averaging_list(wv, text_list):
    """ Take word averaging into a list

    :param wv:
    :param text_list:

    :return:
    """
    return np.vstack([word_averaging(wv, post) for post in text_list])


def w2v_tokenise_text(text):
    """ Tokenise text being passed in using NLTK tokeniser

    :param text: Text to tokenise

    :return: tokens returned from text
    """
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


def label_sentences(corpus, label_type):
    """ Gensim's implementation requires each document/paragraph to have a label associated to it.
        Will do via the TaggedDocument method. Format will be TRAIN_i and TEST_i, where i is
        the dummy index of the text

    :param corpus: The document/paragraph to label
    :param label_type: The label we are assigning to documents/paragraphs

    :return: list of documents/paragraphs with labels
    """
    labelled = []

    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labelled.append(TaggedDocument(v.split(), [label]))

    return labelled


def get_vectors(model, corpus_size, vector_size, vector_type):
    """ Get vectors from trained doc2vec model

    :param model: Trained doc2vec model
    :param corpus_size: Size of data
    :param vector_size: Size of embedding vectors
    :param vectors_type: Training or Testing vectors

    :return: List of vectors
    """

    vectors = np.zeros((corpus_size, vector_size))
    for i in range(0, corpus_size):
        prefix = vector_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]

    return vectors