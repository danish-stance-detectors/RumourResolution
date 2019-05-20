from feature_extraction.word_embeddings import avg_word_emb
from feature_extraction.Annotation import RedditDataset
from feature_extraction.afinn_sentiment import get_afinn_sentiment
from feature_extraction.polyglot_pos import pos_tags_occurence
import re

# Module for extracting features from comment annotations

sarcasm_token = re.compile(r'/[sS][^A-ZÆØÅa-zæøå0-9]')

class FeatureExtractor:

    def __init__(self, dataset, test=False):
        # using passed annotations if not testing
        if test:
            self.dataset = RedditDataset()
        else:
            self.dataset = dataset
        self.bow_words = set()
        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }
    
    def create_feature_vector_test(self, annotation):
        self.dataset.add_annotation(annotation)
        return self.create_feature_vector(annotation, False, False, False, False, False, False, False, False)

    def create_feature_vectors(self, data, text, lexicon, sentiment, reddit, most_freq, bow, pos, wembs, live):
        feature_vectors = []
        for annotation in data:
            instance = self.create_feature_vector(
                annotation, text, lexicon, sentiment, reddit, most_freq, bow, pos, wembs, live
            )
            feature_vectors.append(instance)
        return feature_vectors

    # Extracts features from comment annotation and extends the different kind of features to eachother.
    def create_feature_vector(self, comment, text, lexicon, sentiment, reddit, most_freq, bow, pos, wembs, live):
        feature_vec = list()
        if text:
            feature_vec.append(self.text_features(comment.text, comment.tokens))
        if lexicon:
            feature_vec.append(self.special_words_in_text(comment.tokens, comment.text))
        if sentiment:
            feature_vec.append(self.normalize(get_afinn_sentiment(comment.text), 'afinn_score'))
        if reddit:
            feature_vec.append(self.reddit_comment_features(comment))
        if most_freq:
            feature_vec.append(self.most_frequent_words_for_label(comment.tokens, most_freq))
        if bow:
            feature_vec.append(self.get_bow_presence(comment.tokens))
        if pos:
            feature_vec.append(pos_tags_occurence(comment.text))
        if wembs:
            word_embs = [comment.sim_to_src, comment.sim_to_prev, comment.sim_to_branch]
            avg_wembs = avg_word_emb(comment.tokens)
            word_embs.extend(avg_wembs)
            feature_vec.append(word_embs)
        
        if not live:
            parent_sdqc = self.sdqc_to_int[comment.sdqc_parent]
            sub_sdqc = self.sdqc_to_int[comment.sdqc_submission]
            
            return comment.comment_id, parent_sdqc, sub_sdqc, feature_vec
        else: # avoid annotations if live
            return feature_vec

    def text_features(self, text, tokens):
        # **Binary occurrence features**
        period = int('.' in text)
        e_mark = int('!' in text)
        q_mark = int('?' in text or any(word.startswith('hv') for word in text.split()))
        hasTripDot = int('...' in text)

        # **(Normalized) count features**
        txt_len = self.normalize(len(text), 'txt_len') if len(text) > 0 else 0
        url_count = self.normalize(tokens.count('urlurlurl'), 'url_count')
        # longest sequence of capital letters, default empty for 0 length
        cap_sequence_max_len = len(max(re.findall(r"[A-ZÆØÅ]+", text), key=len, default=''))
        cap_sequence_max_len = self.normalize(cap_sequence_max_len, 'cap_sequence_max_len')
        tripDotCount = self.normalize(text.count('...'), 'tripDotCount')
        q_mark_count = self.normalize(text.count('?'), 'q_mark_count')
        e_mark_count = self.normalize(text.count('!'), 'e_mark_count')
        # Ratio of capital letters
        cap_count = self.normalize(sum(1 for c in text if c.isupper()), 'cap_count')
        cap_ratio = float(cap_count) / float(len(text)) if len(text) > 0 else 0.0
        # number of words
        tokens_len = 0
        avg_word_len = 0
        if len(tokens) > 0:
            tokens_len = self.normalize(len(tokens), 'tokens_len')
            avg_word_len_true = sum([len(word) for word in tokens]) / len(tokens)
            avg_word_len = self.normalize(avg_word_len_true, 'avg_word_len')
        return [period, e_mark, q_mark, hasTripDot, url_count, tripDotCount, q_mark_count,
                e_mark_count, cap_ratio, txt_len, tokens_len, avg_word_len, cap_sequence_max_len]

    def special_words_in_text(self, tokens, text):
        swear_count = self.count_lexicon_occurence(tokens, self.dataset.swear_words)
        negation_count = self.count_lexicon_occurence(tokens, self.dataset.negation_words)
        positive_smiley_count = self.count_lexicon_occurence(text.split(), self.dataset.positive_smileys)
        negative_smiley_count = self.count_lexicon_occurence(text.split(), self.dataset.negative_smileys)

        return [
            self.normalize(swear_count, 'swear_count'),
            self.normalize(negation_count, 'negation_count'),
            self.normalize(positive_smiley_count, 'positive_smiley_count'),
            self.normalize(negative_smiley_count, 'negative_smiley_count')]

    def reddit_comment_features(self, comment):
        karma_norm = self.normalize(comment.user_karma, 'karma')
        edited = int('edit:' in comment.text.lower())
        sarcasm = 1 if sarcasm_token.search(comment.text) else 0
        upvotes_norm = self.normalize(comment.upvotes, 'upvotes')
        reply_count_norm = self.normalize(comment.reply_count, 'reply_count')
        return [karma_norm, int(comment.user_gold_status), int(comment.user_is_employee),
                int(comment.user_has_verified_email), upvotes_norm, reply_count_norm,
                int(comment.is_submitter), edited, sarcasm]

    def most_frequent_words_for_label(self, tokens, n_most):
        vec = []
        histograms = self.dataset.get_frequent_words(n_most)
        for sdqc_id, histogram in histograms.items():
            for freq_token in histogram:
                vec.append(int(freq_token in tokens))
        return vec

    # Gets BOW presence (binary) for tokens
    def get_bow_presence(self, tokens):
        return [1 if w in tokens else 0 for w in self.dataset.bow]

    ### HELPER METHODS ###

    # Counts the amount of words which appear in the lexicon
    def count_lexicon_occurence(self, words, lexion):
        return sum([1 if word in lexion else 0 for word in words])

    def normalize(self, x_i, prop):
        if x_i == 0:
            return 0
        min_x = self.dataset.get_min(prop)
        max_x = self.dataset.get_max(prop)
        if max_x-min_x != 0:
            return (x_i-min_x)/(max_x-min_x)
        
        return x_i

    ### END OF HELPER METHODS ###