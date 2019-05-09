from nltk import word_tokenize
import re, copy, random
from sklearn.model_selection import train_test_split
from feature_extraction.afinn_sentiment import get_afinn_sentiment
import feature_extraction.word_embeddings

url_tag = 'urlurlurl'
regex_url = re.compile(
    r"([(\[]?(https?://)|(https?://www.)|(www.))(?:[a-zæøåA-ZÆØÅ]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')
quote_tag = 'refrefref'
regex_quote = re.compile(r">(.+?)\n")

rand = random.Random(42)

class RedditAnnotation:
        
    # initialises comment annotation class given json
    def __init__(self, json, is_source=False, test=False, live=False):
        self.is_source = is_source

        if test:
            self.comment_id = "test"
            self.text = json
            self.tokens = word_tokenize(json.lower())
            
            # sdcq is just placeholder values
            self.sdqc_parent = "Supporting"
            self.sdqc_submission = "Supporting"
            return
        
        if live:
            comment_json = json
        else:
            comment_json = json["comment"] if not is_source else json
            
        self.text = comment_json["text"]
        self.text = self.filter_reddit_quotes(self.text)
        self.text = self.filter_text_urls(self.text)

        self.sim_to_src = 0
        self.sim_to_prev = 0
        self.sim_to_branch = 0
        if is_source:
            self.comment_id = comment_json["submission_id"]
            self.title = json["title"]
            self.num_comments = json["num_comments"]
            self.url = json["url"]
            self.text_url = json["text_url"]
            self.is_video = json["is_video"]
            # self.subreddit = json["subreddit"] # irrelevant
            # self.comments = json["comments"] # irrelevant
            self.reply_count = comment_json["num_comments"]
            self.is_submitter = True
            self.is_rumour = json["IsRumour"]
            self.is_irrelevant = json["IsIrrelevant"]
            self.truth_status = json["TruthStatus"]
            self.rumour = json["RumourDescription"]
            sdqc_source = json["SourceSDQC"]
            sdqc = "Commenting" if sdqc_source == "Underspecified" else sdqc_source
            self.sdqc_parent = sdqc
            self.sdqc_submission = sdqc
            self.tokens = self.tokenize(self.title)
        else:
            # comment specific info
            self.comment_id = comment_json["comment_id"]
            self.parent_id = comment_json["parent_id"]
            self.comment_url = comment_json["comment_url"]
            self.is_submitter = comment_json["is_submitter"]
            self.is_deleted = comment_json["is_deleted"]
            self.reply_count = comment_json["replies"]
            self.tokens = self.tokenize(self.text)

            # annotation info
            if not live: # live have not been annotated
                self.annotator = json["annotator"]
                self.sdqc_parent = comment_json["SDQC_Parent"]
                self.sdqc_submission = comment_json["SDQC_Submission"]
                self.certainty = comment_json["Certainty"]
                self.evidentiality = comment_json["Evidentiality"]
                self.annotated_at = comment_json["AnnotatedAt"]

        # general info
        self.submission_id = comment_json["submission_id"]
        self.created = comment_json["created"]
        self.upvotes = comment_json["upvotes"]

        # user info
        if "id" in comment_json["user"]: # to skip deleted users
            self.user_id = comment_json["user"]["id"]
            self.user_name = comment_json["user"]["username"]
            self.user_created = comment_json["user"]["created"]
            self.user_karma = comment_json["user"]["karma"]
            self.user_gold_status = comment_json["user"]["gold_status"]
            self.user_is_employee = comment_json["user"]["is_employee"]
            self.user_has_verified_email = comment_json["user"]["has_verified_email"]
        else: # default values
            self.user_id = 'deleted'
            self.user_name = 'deleted'
            self.user_created = '1970-01-01 00:00:00'
            self.user_karma = 0
            self.user_gold_status = False
            self.user_is_employee = False
            self.user_has_verified_email = False

    def tokenize(self, text):
        # Convert all words to lower case and tokenize
        text_tokens = word_tokenize(text.lower(), language='danish')
        tokens = []
        # Remove non-alphabetic characters, not contained in abbreviations
        for token in text_tokens:
            if not punctuation.match(token):
                tokens.append(token)
        return tokens

    def filter_reddit_quotes(self, text):
        """filters text of all annotations to replace reddit quotes with 'refrefref'"""
        return regex_quote.sub(quote_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace 'URLURLURL'"""
        return regex_url.sub(url_tag, text)

class RedditSubmission:
    def __init__(self, source):
        self.source = source
        self.branches = []

    def add_annotation_branch(self, annotation_branch):
        """Add a branch as a list of annotations to this submission"""
        self.branches.append(annotation_branch)


def compute_similarity(annotation, previous, source, branch_tokens, is_source=False):
    # TODO: exclude itself???
    annotation.sim_to_branch = word_embeddings.cosine_similarity(annotation.tokens, branch_tokens)
    if not is_source:
        annotation.sim_to_src = word_embeddings.cosine_similarity(annotation.tokens, source.tokens)
        annotation.sim_to_prev = word_embeddings.cosine_similarity(annotation.tokens, previous.tokens)

def read_lexicon(file_path):
    """Loads lexicon file given path. Assumes file has one word per line"""
    with open(file_path, "r", encoding='utf8') as lexicon_file:
        return set([line.strip().lower() for line in lexicon_file.readlines()])

def count_lexicon_occurence(words, lexion):
    return sum([1 if word in lexion else 0 for word in words])

class RedditDataset:
    def __init__(self):
        self.annotations = {}
        self.anno_to_branch_tokens = {}
        self.anno_to_prev = {}
        self.anno_to_source = {}
        self.submissions = []
        self.last_submission = lambda: len(self.submissions) - 1
        # mapping from property to tuple: (min, max)
        self.min_max = {
            'karma': [0, 0],
            'txt_len': [0, 0],
            'tokens_len': [0, 0],
            'avg_word_len': [0, 0],
            'upvotes': [0, 0],
            'reply_count': [0, 0],
            'afinn_score': [0,0],
            'url_count': [0, 0],
            'quote_count': [0, 0],
            'cap_sequence_max_len': [0, 0],
            'tripDotCount': [0, 0],
            'q_mark_count': [0, 0],
            'e_mark_count': [0, 0],
            'cap_count': [0, 0],
            'swear_count': [0, 0],
            'negation_count': [0, 0],
            'positive_smiley_count': [0, 0],
            'negative_smiley_count': [0, 0]
        }
        self.min_i = 0
        self.max_i = 1
        self.karma_max = 0
        self.karma_min = 0
        # dictionary at idx #num is used for label #num, example: support at 0
        self.freq_histogram = [dict(), dict(), dict(), dict()]
        self.unique_freq_histogram = {}
        self.bow = set()
        self.freq_tri_gram = [dict(), dict(), dict(), dict()]
        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }

        self.positive_smileys = read_lexicon('./data/lexicon/positive_smileys.txt')
        self.negative_smileys = read_lexicon('./data/lexicon/negative_smileys.txt')
        self.swear_words = read_lexicon('./data/lexicon/swear_words.txt')
        swear_words_en = read_lexicon('./data/lexicon/swear_words_en.txt')
        for word in swear_words_en:
            self.swear_words.add(word)
        self.negation_words = read_lexicon('./data/lexicon/negation_words.txt')

    def add_annotation(self, annotation):
        """Add to self.annotations. Should only be uses for testing purposes"""
        annotation = self.analyse_annotation(annotation)
        if annotation.comment_id not in self.annotations:
            self.annotations[annotation.comment_id] = annotation

    def add_reddit_submission(self, source):
        self.submissions.append(RedditSubmission(RedditAnnotation(source, is_source=True)))

    def add_submission_branch(self, branch, sub_sample=False):
        annotation_branch = []
        branch_tokens = []
        class_comments = 0
        # First, convert to Python objects
        for annotation in branch:
            annotation = RedditAnnotation(annotation)
            if self.sdqc_to_int[annotation.sdqc_submission] == 3:
                class_comments += 1
            branch_tokens.extend(annotation.tokens)
            annotation_branch.append(annotation)

        # Filter out branches with pure commenting class labels
        if sub_sample and class_comments == len(branch):
            return

        # Compute cosine similarity
        source = self.submissions[self.last_submission()].source
        prev = source
        for annotation in annotation_branch:
            if annotation.comment_id not in self.annotations:  # Skip repeated annotations
                compute_similarity(annotation, prev, source, branch_tokens)
                self.analyse_annotation(annotation)  # Analyse relevant annotations
                self.annotations[annotation.comment_id] = annotation
                self.anno_to_branch_tokens[annotation.comment_id] = branch_tokens
                self.anno_to_prev[annotation.comment_id] = prev
                self.anno_to_source[annotation.comment_id] = source
            prev = annotation
        self.submissions[self.last_submission()].add_annotation_branch(annotation_branch)  # This might be unnecessary

    def train_test_split(self, test_size=0.25, rand_state=42, shuffle=True, stratify=True):
        X = []
        y = []
        for annotation in self.iterate_annotations():
            X.append(annotation)
            y.append(self.sdqc_to_int[annotation.sdqc_submission])
        print('Splitting with test size', test_size)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=test_size, random_state=rand_state, shuffle=shuffle, stratify=(y if stratify else None)
        )
        print('Train stats:')
        self.print_status_report(X_train)
        print('Test stats:')
        self.print_status_report(X_test)
        return X_train, X_test


    def print_status_report(self, annotations=None):
        histogram = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        ix_to_sdqc = {0: 'S', 1:'D', 2: 'Q', 3: 'C'}
        n = 0
        for annotation in (self.iterate_annotations() if not annotations else annotations):
            histogram[self.sdqc_to_int[annotation.sdqc_submission]] += 1
            n += 1
        print('Number of data points:', n)
        print('SDQC distribution:')
        for label, count in histogram.items():
            print('{}: {:4d} ({:.3f})'.format(ix_to_sdqc[label], count, float(count)/float(n)))

    def analyse_annotation(self, annotation):
        if not annotation:
            return
        self.handle(self.min_max['karma'], annotation.user_karma)
        self.handle(self.min_max['txt_len'], len(annotation.text))
        self.handle(self.min_max['afinn_score'], get_afinn_sentiment(annotation.text))
        self.handle(self.min_max['url_count'], annotation.tokens.count('urlurlurl'))
        self.handle(self.min_max['quote_count'], annotation.tokens.count('refrefref'))
        self.handle(self.min_max['cap_sequence_max_len'], len(max(re.findall(r"[A-ZÆØÅ]+", annotation.text), key=len, default='')))
        self.handle(self.min_max['tripDotCount'], annotation.text.count('...'))
        self.handle(self.min_max['q_mark_count'], annotation.text.count('?'))
        self.handle(self.min_max['e_mark_count'], annotation.text.count('!'))
        self.handle(self.min_max['cap_count'], sum(1 for c in annotation.text if c.isupper()))
        self.handle(self.min_max['swear_count'], count_lexicon_occurence(annotation.tokens, self.swear_words))
        self.handle(self.min_max['negation_count'], count_lexicon_occurence(annotation.tokens, self.negation_words))
        self.handle(self.min_max['positive_smiley_count'], count_lexicon_occurence(annotation.text.split(),
                                                                                   self.positive_smileys))
        self.handle(self.min_max['negative_smiley_count'], count_lexicon_occurence(annotation.text.split(),
                                                                                   self.negative_smileys))
        
        word_len = len(annotation.tokens)
        if not word_len == 0:
            self.handle(self.min_max['tokens_len'], word_len)
            self.handle(self.min_max['avg_word_len'],
                        sum([len(word) for word in annotation.tokens]) / word_len)
        self.handle(self.min_max['upvotes'], annotation.upvotes)
        self.handle(self.min_max['reply_count'], annotation.reply_count)
        # self.handle_frequent_words(annotation) # skip frequent, we dont have labels here
        self.handle_bow(annotation.tokens)
        # self.handle_ngram(annotation, self.freq_tri_gram, 3) # skil ngrams, we dont have labels here

        return annotation

    def handle(self, entries, prop):
        if prop > entries[self.max_i]:
            entries[self.max_i] = prop
        if prop < entries[self.min_i] or entries[self.min_i] == 0:
            entries[self.min_i] = prop

    def get_min(self, key):
        return self.min_max[key][self.min_i]

    def get_max(self, key):
        return self.min_max[key][self.max_i]

    def handle_frequent_words(self, annotation, use_parent_sdqc=False):
        # Most frequent words for annotation classes, string to int (word counter)
        dict_idx = self.sdqc_to_int[annotation.sdqc_parent] \
            if use_parent_sdqc else self.sdqc_to_int[annotation.sdqc_submission]
        for token in annotation.tokens:
            current_histo = self.freq_histogram[dict_idx]
            if token in current_histo:
                current_histo[token] = current_histo[token] + 1
            else:
                current_histo[token] = 1

    def handle_bow(self, annotation_tokens):
        for t in annotation_tokens:
            self.bow.add(t)

    def handle_ngram(self, annotation, gram_dict, gram_size):
        annotation_tokens = annotation.tokens
        label = self.sdqc_to_int[annotation.sdqc_submission]
        label_dict = gram_dict[label]
        for (idx, t) in enumerate(annotation_tokens):
            if idx + gram_size < len(annotation_tokens)-1:
                seq = " ".join(annotation_tokens[idx:idx+gram_size])
                if seq in label_dict:
                    label_dict[seq] = label_dict[seq] + 1
                else:
                    label_dict[seq] = 1
        

    def get_frequent_words(self, take_count):
        if self.unique_freq_histogram:
            return self.unique_freq_histogram
        histogram = {}
        word_count = {}
        for idx in range(len(self.freq_histogram)):
            keys = [(self.freq_histogram[idx][key], key) for key in self.freq_histogram[idx].keys()]
            keys.sort()
            keys.reverse()

            histogram[idx] = keys[:take_count]

            for (freq, word) in histogram[idx]:
                if word in word_count:
                    word_count[word] = word_count[word] + 1
                else:
                    word_count[word] = 1

        unique_histograms = {0: [], 1: [], 2: [], 3: []}

        for key, values in histogram.items():
            for (freq, word) in values:
                if word_count[word] == 4:
                    continue
                unique_histograms[key].append(word)

        self.unique_freq_histogram = unique_histograms

        return unique_histograms

    def iterate_annotations(self):
        for anno_id, annotation in self.annotations.items():
            yield annotation

    def iterate_branches(self, with_source=True):
        for submission in self.submissions:
            for branch in submission.branches:
                if with_source:
                    yield submission.source, branch
                else:
                    yield branch

    def iterate_submissions(self):
        for submission in self.submissions:
            yield submission

    def size(self):
        return len(self.annotations)
