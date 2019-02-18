import logging


def get_logger():
    logger = logging.getLogger('logger')
    log_formatter = '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s : %(message)s'

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    return logger


log = get_logger()
