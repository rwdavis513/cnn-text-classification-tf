import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import FLAGS
from tensorflow.contrib import learn


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_movie_reviews(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_accounting_data(data_file_path):
    """
    Loads Accounting Transactions in a csv format. Using the "Description" column as the input data and the
      "category_name" column as the response.
    :param data_file_path: full path to a data file
    :return: two item list with a list of the Descriptions for X and a list of the corresponding category names
    """
    df = pd.read_csv(data_file_path)
    df = df[~df['description'].isnull()]
    print("Raw data shape: {}".format(df.shape))
    x = df['description'].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df['category_name'].tolist())
    y = y.reshape((y.shape[0], 1))   # Transform from (?, ) to (?, 1)
    return x, y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def calculate_max_length(x_text):
    """
    Caculates the maximum length
    :param x_text:
    :return:
    """
    x_lengths = []
    for x in x_text:
        try:
            length = len(x.split(" "))
        except AttributeError as e:
            print("error: {}".format(e))
            print(x)
            length = 1
        x_lengths.append(length)
    return max(x_lengths)


def load_data(train_or_eval=None):
    # Load data
    print("Loading data...")
    if FLAGS.accounting_data_file:
        print("Loading accounting data...")
        x_text, y = load_accounting_data(FLAGS.accounting_data_file)
    else:
        print("Loading movie reviews...")
        x_text, y = load_movie_reviews(FLAGS.positive_data_file, FLAGS.negative_data_file)

    print("x_text: {} y: {}".format(len(x_text), len(y)))

    if train_or_eval == 'evaluation':
        return x_text, y

    # Build vocabulary
    max_document_length = calculate_max_length(x_text)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    print("x: {} y: {}".format(x.shape, len(y)))
    # Split train/test set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y)
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return (x_train, x_dev, y_train, y_dev), vocab_processor


def get_num_classes(y_train):
    """
    Hack to check if there is only one class.
    :param y_train:
    :return:
    """
    try:
        return y_train.shape[1]
    except KeyError:
        return 1
    except IndexError:
        print(y_train)
        return 1
