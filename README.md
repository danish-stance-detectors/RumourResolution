# Reddit Veracity Classification

This is a tool which can guess whether a rumourous reddit submission in danish is true or false.

It applies stance classification and then rumour veracity classification on the stance labels.

## Prerequisites

### Python libraries
The tool requires python and a number of libraries to be installed:

* Afinn
* Numpy
* scikit learn
* hmmlearn
* nltk
* psaw
* praw
* joblib
* ...

### Reddit Permissions
For this tool to work, a file 'praw.ini' must be created in this folder.

It should have the format presented below:

[uuuu]
client_id=XXX
client_secret=XXX
user_agent=python:XXX:v1.0 (by /u/<Reddit_user_name>)

Where u is the name used on the command line when calling the program.
The application, client_id and client_secret can be obtained by following [these steps](https://github.com/reddit-archive/reddit/wiki/OAuth2).
Please note the username of your own account must replace the '<Reddit_user_name>'.

### Danish word embeddings

Danish word2vec word embeddings must be downloaded and added to '/data/word_embeddings/' folder.

They can be obtained [here](https://figshare.com/articles/Danish_DSL_and_Reddit_word2vec_word_embeddings/8099927).

## Running the tool

To run the tool run 'py veracity.py -u <uuuu> -s_id <submissionID>'

Where uuuu should match the [uuuu] in the praw.ini file and the submissionID should match the reddit submission you want to analyse.

## Credits

* [DSL](https://dsl.dk/)
...The word embeddings have been trained on both sentence data from [dsl](https://dsl.dk/) and on reddit data from the danish stance dataset.
* [Afinn](https://github.com/fnielsen/afinn)
...The afinn sentiment is facilitated by the afinn sentiment library, which has been linked above. Further credits can be seen below.
...Finn Årup Nielsen, "A new ANEW: evaluation of a word list for sentiment analysis in microblogs", Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages. Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)