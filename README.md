# Reddit Veracity Classification

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

## Running the tool

To run the tool run 'py veracity.py -u <uuuu> -s_id <submissionID>'

Where uuuu should match the [uuuu] in the praw.ini file and the submissionID should match the reddit submission you want to analyse.

