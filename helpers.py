"""
" helpers.py
"
" This file contains helper function(s).
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 17 May, 2020
:Version: 1.0.0
"""


def d_types(type_name):
    """
    Helper function to get right data type names for both implementations.

    @param type_name: (str) Data type's general name.
    @return (tuple[str][str]) Corresponding data type's name that Numpy and
        CUDA C uses.
    """    

    if type_name == 'float':
        np_type = 'float32'
        c_type = 'float'
    elif type_name == 'double':
        np_type = 'float64'
        c_type = 'double'
    elif type_name == 'int':
        np_type = 'int32'
        c_type = 'int'

    return (np_type, c_type)
