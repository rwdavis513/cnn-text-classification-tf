from data_helpers import load_accounting_data, load_data
import config
import os

ACCOUNTING_DATA_FILE_PATH = os.path.join(config.DATA_DIR, 'line-items-5000.csv')


def test_load_accounting_data():
    xy_list = load_accounting_data(ACCOUNTING_DATA_FILE_PATH)
    assert type(xy_list) == list
    x, y = xy_list
    assert len(x) == len(y)


def test_load_data():
    (x_train, x_dev, y_train, y_dev), vocab_processor = load_data('accounting_data')