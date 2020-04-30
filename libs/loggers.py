import logging
import os

import torch

prepare_logger, final_logger, cuda_logger = None, None, None
train_logger, validation_logger, loss_logger = None, None, None
sample_logger = None


def get_logger(name, log_file, level=logging.INFO, clear=True):
    if clear:
        open(log_file, 'w').close()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for hand in logger.handlers:
        logger.removeHandler(hand)
    logger.addHandler(handler)
    return logger


def log_info(logger, msg):
    print(msg)
    if logger is not None:
        logger.info(msg)


def cuda_mem_in_mb():
    return torch.cuda.memory_allocated() / 2 ** 20


def initial_loggers(log_path, clear=True):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    global prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger, sample_logger
    if not (None not in (
            prepare_logger, train_logger, validation_logger, final_logger, cuda_logger, loss_logger, sample_logger)):
        prepare_logger = get_logger('prepare logger', log_path + 'pre.log', clear=clear)
        train_logger = get_logger('gpt2 train logger', log_path + 'train.log', clear=clear)
        validation_logger = get_logger('gpt2 validation logger', log_path + 'val.log', clear=clear)
        final_logger = get_logger('final logger', log_path + 'fin.log', clear=clear)
        cuda_logger = get_logger('cuda logger', log_path + 'cuda.log', clear=clear)
        loss_logger = get_logger('loss logger', log_path + 'los.log', clear=clear)
        sample_logger = get_logger('sample logger', log_path + 'sample.log', clear=clear)
