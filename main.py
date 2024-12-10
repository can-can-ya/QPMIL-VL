from utils import get_args, get_config, init, print_config
from manager import Manager


def main(config):
    print_config(config)
    manager = Manager(config) # Operation Class
    manager.incre_train() # incremental training
    print('[infor] end of program execution')


if __name__ == '__main__':
    cfg = get_args() # configuration parameters from the command line
    config = get_config(cfg['config']) # configuration parameters from the config file
    init(cfg, config)
    main(config)