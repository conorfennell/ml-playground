"""
json support
"""
from __future__ import absolute_import
import json

def write_json_to_file(json_data, file_name):
    """Write json object to file"""
    with open(file_name, 'w') as out_file:
        json.dump(json_data, out_file)

def read_json_file(file_name):
    """Read json from a file"""
    with open(file_name) as data_file:
        return json.load(data_file)

