import sys, os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
import argparse

word2vec_path = '../data/word_embeddings/word2vec/'
fasttext_path = '../data/fasttext/'
dsl_sentences = '../data/corpus/dsl_sentences.txt'
reddit_sentences = '../data/corpus/reddit_sentences.txt'
wiki_sentences = '../../Data/Wiki_Corpus/wiki_sentences.txt'
datafolder = './data/word_embeddings/'
fasttext_model = os.path.join(datafolder, 'fasttext/cc.da.300.bin')
fasttext_data = os.path.join(datafolder, 'fasttext/fasttext_dsl_sentences_reddit_sentences_300_cbow_negative.kv')
word2vec_data = lambda dim: os.path.join(datafolder,
                                         'word2vec_dsl_sentences_reddit_sentences_{0}_cbow_negative.kv'.format(dim))
save_path = lambda algo: word2vec_path if algo == 'word2vec' else fasttext_path

wv_model = None
vector_size = 300


# memory friendly iterator
class MySentences:
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf8') as corpus:
                for line in corpus:
                    line = line.rstrip('\n')
                    if line:
                        yield line.split()

    def __len__(self):
        n = 0
        for filename in self.filenames:
            with open(filename, 'r', encoding='utf8') as corpus:
                n += len(corpus.readlines())
        return n


def train_save_word_embeddings(corpus_file_path, algo, vector_dim, word2vec_format=False, save_model=False,
                               architecture='cbow', train_algorithm='negative', workers=4):
    """architecture: 'skim-gram' or 'cbow'. train_algorithm: 'softmax' or 'negative'"""
    sentences = MySentences(corpus_file_path)
    arch = 1 if architecture=='skip-gram' else 0
    train = 1 if train_algorithm=='softmax' else 0
    print('Training %s with size %d' % (algo, vector_dim))
    if algo == 'word2vec':
        model = Word2Vec(sentences=sentences, size=vector_dim, workers=workers, sg=arch, hs=train)
    else:  # fasttext
        model = FastText(sentences=sentences, size=vector_dim, workers=workers, sg=arch, hs=train)
    print('Done!')
    s = algo + '_'
    for name in corpus_file_path:
        s += name.split('/')[-1].split('.')[0] + '_'
    filename = "{0}{1}_{2}_{3}".format(s, vector_dim, architecture, train_algorithm)
    if save_model:
        print('Saving model in {0}.model'.format(filename))
        model.save(os.path.join(word2vec_path, "{}.model".format(filename)))
    if word2vec_format:
        print('Saving word embeddings in original C word2vec (.txt) format in {}.txt'.format(filename))
        model.wv.save_word2vec_format(os.path.join(word2vec_path, "{}.txt".format(filename)))
    print('Saving word embeddings in {0}.kv'.format(filename))
    model.wv.save(os.path.join(save_path(algo), "{}.kv".format(filename)))
    print('Saved!')
    return model


def save_fasttext(path_to_vectors, saved_filename):
    model = load_word_embeddings_bin(path_to_vectors)
    print('Saving word embeddings')
    model.wv.save(os.path.join(fasttext_path, saved_filename))
    print('Done!')
    

def load_saved_word_embeddings(w2v, fasttext):
    global wv_model
    if w2v:
        wv_model = KeyedVectors.load(word2vec_data(w2v))
        global vector_size
        vector_size = w2v
    elif fasttext:
        wv_model = KeyedVectors.load(fasttext_data)
    return wv_model


def load_word_embeddings_bin(filename, algorithm='fasttext'):
    print('loading model...')
    global wv_model
    if(algorithm == 'fasttext'):
        wv_model = FastText.load_fasttext_format(filename)
    elif(algorithm == 'word2vec'):
        wv_model = KeyedVectors.load_word2vec_format(filename, encoding='utf8', binary=True)
    print('Done!')
    return wv_model


def load_and_train_fasttext(corpus_file_path):
    ft_model = load_word_embeddings_bin(fasttext_model)
    sentences = MySentences(corpus_file_path)
    print('Building vocab')
    ft_model.build_vocab(sentences, update=True)
    print('Done')
    print('Training...')
    ft_model.train(sentences=sentences, total_examples=sentences.__len__(), epochs=ft_model.epochs)
    print('Done')
    print('Saving word vectors')
    ft_model.wv.save(os.path.join(fasttext_path, 'fasttext_da_300_dsl_reddit.kv'))
    print('Done')


def avg_word_emb(tokens):
    global wv_model
    if not wv_model:
        return None
    vec = np.zeros(vector_size)  # word embedding
    # make up for varying lengths with zero-padding
    n = len(tokens)
    if n == 0:
        return vec.tolist()
    for w_i in range(n):
        token = tokens[w_i]
        if token in wv_model.vocab:
            vec += wv_model[token]
    # Average word embeddings
    return (vec/n).tolist()


def cosine_similarity(one, other):
    global wv_model
    if not wv_model:
        return None

    # Lookup words in w2c vocab
    words = []
    for token in one:
        if token in wv_model.vocab:  # check that the token exists
            words.append(token)
    other_words = []
    for token in other:
        if token in wv_model.vocab:
            other_words.append(token)

    if len(words) > 0 and len(other_words) > 0:  # make sure there is actually something to compare
        # cosine similarity between two sets of words
        return wv_model.n_similarity(other_words, words)
    else:
        return 0.  # no similarity if one set contains 0 words


most_sim_cache = {}
def most_similar_word(word):
    if word in most_sim_cache:
        return most_sim_cache[word]
    global wv_model
    if wv_model and word in wv_model.vocab:
        most_sim = wv_model.similar_by_word(word)
        most_sim_cache[word] = most_sim
        return most_sim
    return [(word, 1)]


word_sim_cache = {}
def word_vector_similarity(w1, w2):
    if (w1, w2) in word_sim_cache:
        return word_sim_cache[(w1, w2)]
    if (w2, w1) in word_sim_cache:
        return word_sim_cache[(w2, w1)]
    global wv_model
    if wv_model and w1 in wv_model.vocab and w2 in wv_model.vocab:
        sim = wv_model.similarity(w1, w2)
        word_sim_cache[(w1, w2)] = sim
        word_sim_cache[(w2, w1)] = sim
        return sim
    return 0


def in_vocab(word):
    if wv_model and word in wv_model.vocab:
        return True
    return False


def main(argv):
    # arguments setting 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_save', type=str, nargs=1,
                        help='Train and save word vectors with "word2vec" or "fasttext"')
    parser.add_argument('-v', '--vector_size', type=int, default=300, help='the size of a word vector')
    parser.add_argument('--architecture', type=str, default='cbow', help='the architecture: "skip-gram" or "cbow"')
    parser.add_argument('--train_algorithm', type=str, default='negative', help='the training algorithm: "softmax" or "negative"')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--word2vec_format', action='store_true', default=False, help='Store in the original C word2vec (.txt) format')
    parser.add_argument('--fasttext_load_train', action='store_true', default=False,
                        help='Train a fastText model on a corpus (Default: Reddit corpus) and save word vectors')
    args = parser.parse_args(argv)

    vector_dim = args.vector_size
    architecture = args.architecture
    train_algorithm = args.train_algorithm
    word2vec_format = args.word2vec_format
    workers = args.workers
    if args.train_save:
        train_save_word_embeddings([dsl_sentences, reddit_sentences], algo=args.train_save[0],
                                   vector_dim=vector_dim, word2vec_format=word2vec_format,
                                   architecture=architecture, train_algorithm=train_algorithm, workers=workers)
    if args.fasttext_load_train:
        load_and_train_fasttext([dsl_sentences, reddit_sentences])


if __name__ == "__main__":
    main(sys.argv[1:])
