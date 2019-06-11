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
            'reddit' : False,
            'most_freq' : False,
            'bow' : False,
            'pos' : False,
            'wembs' : True 
        }

        # loads bow used for model training and injects it into dataset
        if features['bow']:
            train_bow = []
            with open('./data/annotated_bow.txt', 'r', encoding='utf8') as file:
                for line in file.readlines():
                    train_bow.append(line)
            dataset.bow = train_bow
        
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
        for i in range(len(stance_predicts)):
            print("Post:\n{}\nLabel: {}\n".format(dataset.annotations[i].text, num_to_stance[stance_predicts[i]]))
        # print([num_to_stance[x] for x in stance_predicts])
        hmm_clf = load('./models/hmm_1_branch.joblib') 
        rumour_veracity = hmm_clf.predict([stance_predicts])[0]

        is_true = None
        if rumour_veracity:
            is_true = 'true'
        else:
            is_true = 'false'
        
        print("It seems the crowds stance thinks submission '{}' is {}".format(sub['title'], is_true))

if __name__ == "__main__":
    main(sys.argv[1:])