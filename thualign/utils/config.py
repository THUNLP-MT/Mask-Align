# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import configparser
import logging

def parse(value):
    res = value
    if value[0] in ('[', '(') and value[-1] in (']', ')'):
        value = value.replace("'", '"')
        value = json.loads(value)
        res = [parse(v) if isinstance(v, str) else v for v in value]
    else:
        try:
            res = float(res)
            if value.isdigit():
                res = int(res)
        except ValueError:
            if value.lower() == 'true':
                res = True
            elif value.lower() == 'false':
                res = False
    return res

class Config:

    def __init__(self, config):
        if isinstance(config, configparser.ConfigParser):
            self.config = config
        else:
            raise ValueError("Unknown config type for %s" % type(config))
        self._params = {}
        for k, v in self.config.items():
            for kk, vv in v.items():
                self._params[kk] = parse(vv)

    def __getattr__(self, name):
        if name in self._params:     
            return self._params[name]
        else:
            raise AttributeError("'Config' object has no attribute '%s'" \
                                % name)

    def export(self, filename):
        with open(filename, 'w') as f:
            self.config.write(f)
    
    def override_config(self, filename, field=('model',)):
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        if os.path.exists(filename):
            config.read(filename)
            for f in field:
                for k, v in config[f].items():
                    self._params[k] = parse(v)
        else:
            raise ValueError("Cannot find config file '%s'" % filename)
    
    def __str__(self):
        return json.dumps(self._params, sort_keys=True)

    @classmethod
    def read(cls, cfg, base=None, data=None, model=None, exp='DEFAULT'):
        curdir = os.path.dirname(__file__)
        if not os.path.exists(cfg):
            cfg = os.path.join(curdir, \
                '../configs/user/{}.config'.format(cfg.replace('.config', '')))
        base = base or os.path.join(curdir, '../configs/base.config')
        model = model or os.path.join(curdir, '../configs/model.config')

        base_config = read_config_file(base)
        user_config = read_config_file(cfg, field=exp, keepdict=True)
        user_config = {
            'exp': {**user_config['DEFAULT'], **user_config[exp]}
        }
        model_cls = user_config['exp']['model']

        model_config = read_config_file(model, field=model_cls, keepdict=True)
        model_config = {
            'model': {**model_config['DEFAULT'], **model_config[model_cls]}
        }

        all_config = merge_config(base_config, model_config)
        all_config = merge_config(all_config, user_config)

        return cls(all_config)

def read_config_file(file, field=None, keepdict=False):
    if os.path.exists(file):
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(file)
        config_dict = config_to_dict(config)
        if field is not None:
            assert field in config_dict
            config_dict = {field: config_dict[field], 'DEFAULT': config_dict['DEFAULT']}
        if keepdict:
            return config_dict
        else:
            res = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            res.read_dict(config_dict)
            return res
    else:
        raise FileNotFoundError("No such file or directory '%s'" % file)

def config_to_dict(config):
    res = config._sections
    res['DEFAULT'] = config.defaults()
    return res

def merge_config(base_config, add_config):
    if isinstance(add_config, configparser.ConfigParser):
        add_config = config_to_dict(add_config)
    base_config.read_dict(add_config)
    return base_config

def reverse_data(data_config):
    for section in data_config:
        for option in data_config[section]:
            value = parse(data_config[section][option])
            if isinstance(value, list):
                value[0], value[1] = value[1], value[0]
            data_config[section][option] = str(value)
    return data_config