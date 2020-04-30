import argparse
import json
import logging
import os

from start_module import start_func

logging.getLogger('transformers.tokenization_utils').disabled = True


def main(config_file='model_config.json'):
    import libs
    os.chdir('/'.join(os.path.abspath(__file__).split('/')[:-1]))
    libs.log_info(libs.loggers.prepare_logger, 'Using config {}'.format(config_file))
    with open(config_file, 'r') as f:
        config = json.load(f) if os.path.exists(config_file) and os.path.isfile(config_file) else {}
    models = None
    for k, v in config.items():
        if isinstance(v, list):
            if models is None:
                models = len(v)
            elif models != len(v):
                raise ValueError('Config field {} has wrong length'.format(k))
    models = models if models is not None else 1
    for i in range(models):
        new_config = {}
        for k, v in config.items():
            if isinstance(v, list):
                new_config[k] = v[i]
            else:
                new_config[k] = v
        start_func(new_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    main(args.config)
