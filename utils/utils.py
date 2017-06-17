"""
json support
"""
from __future__ import absolute_import
import json
from datetime import datetime

EPOCH = datetime.utcfromtimestamp(0)
END_OF_TIME = 9999999999
INTERVAL_SECONDS = 1800 

def write_json_to_file(json_data, file_name):
    """Write json object to file"""
    with open(file_name, 'w') as out_file:
        json.dump(json_data, out_file)

def read_json_file(file_name):
    """Read json from a file"""
    with open(file_name) as data_file:
        return json.load(data_file)

def unix_time_seconds(dt):
    return (dt - EPOCH).total_seconds()