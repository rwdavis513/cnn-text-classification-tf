from eval import *


def test_load_data_for_eval():
    x_test, y_test = load_data_for_eval()
    assert type(x_test) == np.ndarray
    assert type(y_test) == np.ndarray
    assert x_test.shape[0] > 100

