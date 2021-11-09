import configparser
import os

from utils.constant import CONFIG_DIR


def read_configuration(config_file):
    """
    读取配置文件中的配置项
    :param config_file:
    :return:
    """
    config_file = os.path.join(CONFIG_DIR, config_file)
    cp = configparser.ConfigParser()
    cp.read(config_file)

    return cp
