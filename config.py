import os

try:
    DATA_DIR = os.environ['CNN_TEXT_DATA_DIR']
except KeyError:
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

assert os.path.exists(DATA_DIR)
