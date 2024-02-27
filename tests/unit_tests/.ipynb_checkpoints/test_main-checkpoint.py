"""
Created By  : ...
Created Date: DD/MM/YYYY
Description : ...
"""

import os
import sys


cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/../../src")


def test_dummy(caplog):
    """Test for the main function.
    """
    from main import dummy
    res = dummy('Testing')
    assert 'Testing' in caplog.text
    assert res == 'Testing'
