import argparse
import praw 
import prawcore.exceptions
from psaw import PushshiftAPI
import sys
import json
import numpy as np
from feature_extraction.Annotation import RedditAnnotation
from feature_extraction.Annotation import RedditDataset
from feature_extraction.Features import FeatureExtractor
from feature_extraction.word_embeddings import load_saved_word_embeddings
from models.hmm_veracity import HMM

from sklearn.linear_model import LogisticRegression

from joblib import dump, load

import reddit_fetcher
import data_loader

flatten = lambda l: [item for sublist in l for item in sublist]

test_data_file = './data/training_data/preprocessed_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_test.csv'

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocessing of data files for stance classification')

    parser.add_argument('-s_id', '--submission_id', help='Input reddit submission id')
    parser.add_argument('-u', '--user', help='Input reddit user API key name')
    parser.add_argument('-m', '--model', default='./models/logistic_regression.joblib', help='Path to model to user')

    args = parser.parse_args(argv)

    if args.submission_id and args.user:
        reddit = praw.Reddit(args.user)
        dataset = RedditDataset()
        
        load_saved_word_embeddings(300, False)

        sub = reddit_fetcher.getredditsubmission(reddit, args.submission_id)

        annotations = [RedditAnnotation(comment, live=True) for comment in sub['comments']]
        for anno in annotations:
            dataset.add_annotation(anno)

        features = {
            'text': True,
            'lexicon' : False,
            'sentiment' : True,
            'reddit' : True,
            'most_freq' : False,
            'bow' : False,
            'pos' : False,
            'wembs' : True 
        }

        ## Used to train new models
        # X, y = data_loader.read_stance_data(cols_to_take=['text', 'sentiment', 'reddit', 'word2vec'])
        # X_test, y_test = data_loader.read_stance_data(file_name=test_data_file, cols_to_take=['text', 'sentiment', 'reddit', 'word2vec'])
        # X.extend(X_test)
        # y.extend(y_test)

        # clf = LogisticRegression().fit(X, y)
        
        # dump(clf, './models/logistic_regression.joblib')

        clf = load(args.model)

        extractor = FeatureExtractor(dataset)
        vectors = extractor.create_feature_vectors(annotations, 
                                                   features['text'], 
                                                   features['lexicon'], 
                                                   features['sentiment'], 
                                                   features['reddit'], 
                                                   features['most_freq'],
                                                   features['bow'], 
                                                   features['pos'], 
                                                   features['wembs'],
                                                   True) # is live, to avoid annotations
        
        flattened_vectors = []

        for vec in vectors:
            flat_vec = []
            for group in vec:
                if type(group) == list:
                    flat_vec.extend(group)
                else:
                    flat_vec.append(group)
            flattened_vectors.append(flat_vec)
        
        stance_predicts = clf.predict(flattened_vectors)
        
        num_to_stance = {
            0 : 'Supporting',
            1 : 'Denying',
            2 : 'Querying',
            3 : 'Commenting'
        }

        print("Crowd stance ordered by comment time:\n")
        print([num_to_stance[x] for x in stance_predicts])

        hmm_data, _ = data_loader.get_hmm_data()
        y = [x[0] for x in hmm_data]
        X = [x[1] for x in hmm_data]

        hmm_clf = HMM(1).fit(X, y)
        rumour_veracity = hmm_clf.predict([stance_predicts])[0]

        is_true = None
        if rumour_veracity:
            is_true = 'true'
        else:
            is_true = 'false'
        
        print("It seems the crows stance thinks submission '{}' is {}".format(sub['title'], is_true))

if __name__ == "__main__":
    main(sys.argv[1:])