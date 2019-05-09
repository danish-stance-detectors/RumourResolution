import csv


all_features = ['text','lexicon','sentiment','reddit','most_frequent','bow','pos','word2vec']
training_data_file = './data/training_data/preprocessed_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_train.csv'
test_data_file = './data/training_data/preprocessed_text_lexicon_sentiment_reddit_most_frequent100_bow_pos_word2vec300_train.csv'
hmm_datafile = './data/hmm/preprocessed_hmm_no_branch.csv'

tab='\t'

def read_stance_data(file_name=training_data_file, cols_to_take=all_features):
    
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        line_count = 0
        data_X = []
        data_y = []
        for row in reader:
            line = []
            if line_count > 0:
                for col in cols_to_take:
                    if '[' in row[col]:
                        nums = [float(x) for x in row[col].rstrip(']').lstrip('[').split(',')]
                        line.extend(nums)
                    else:
                        line.append(float(row[col]))
            
                data_X.append(line)
                data_y.append(int(row['sdqc_submission']))
            line_count += 1
        
    # return all rows of the specified columns
    return data_X, data_y

def get_hmm_data(filename=hmm_datafile, delimiter=tab):
    data = []
    max_branch_len = 0
    with open(filename) as file:
        has_header = csv.Sniffer().has_header(file.read(1024))
        file.seek(0) # Rewind
        csvreader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(csvreader) # Skip header row
        for row in csvreader:
            truth_status = int(row[0])
            values = row[1].strip("[").strip("]").split(',')
            instance_vec = [float(i.strip()) for i in values]
            data.append((truth_status, instance_vec))
            max_branch_len = max(max_branch_len, len(instance_vec))
    
    return data, max_branch_len