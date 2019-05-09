# Polyglot requires numpy and libicu-dev, where the latter is only available on  ubuntu/debian linux distributions
# To install on Windows, follow these steps:
# 1. install PyICU.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# 2. install PyCLD2.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# 3. git clone https://github.com/aboSamoor/polyglot
# 4. cd polyglot
# 5. python setup.py install
# Then, to use it for the danish language download the necessary models as such:
# $ polyglot download embeddings2.da pos2.da
# Docs: https://polyglot.readthedocs.io/en/latest/POS.html
from polyglot.text import Text

tag_set = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other"
}
N = 17  # length of tag set

def tag_corpus(corpus_file, output_file):
    with open(corpus_file, 'r', encoding='utf-8') as f, open(output_file, 'w+', encoding='utf-8') as out:
        for line in f.readlines():
            text = Text(line, hint_language_code='da')
            out.write(line + ' ')
            for _, tag in text.pos_tags:
                out.write(tag + ' ')
            out.write('\n')

def pos_tags(text):
    text = Text(text, hint_language_code='da')
    pos_tags = []
    for _, tag in text.pos_tags:
        pos_tags.append(tag)
    return pos_tags

def get_tag_set():
    return tag_set.keys()

def pos_tags_occurence(text):
    tags = pos_tags(text)
    res = [0] * N
    for i, tag in enumerate(get_tag_set()):
        if tag in tags:
            res[i] = 1
    return res

