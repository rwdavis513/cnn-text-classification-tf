from train import session_setup


def test_session_setup():
    objs = session_setup()
    assert len(objs) == 7

