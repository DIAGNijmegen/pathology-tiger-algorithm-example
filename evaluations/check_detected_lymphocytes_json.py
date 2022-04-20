import os, json, jsonschema
from collections import OrderedDict
from pathlib import Path

#here you can check the validity of the created detected-lymphocytes.json

def read_json(path):
    path = Path(path)
    with path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def validate_json(path):
    """ path: the path to the created detected-lymphocytes.json """
    cwd = Path(os.path.dirname(__file__))
    schema_path = cwd/"multiple_points.json"
    from jsonschema import validate
    schema = read_json(schema_path)
    out_data = read_json(path)
    return validate(instance=out_data, schema=schema)