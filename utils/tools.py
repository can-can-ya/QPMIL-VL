import argparse
import yaml
import os.path as osp
import errno
import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from .logger import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help="path to the config file")
    parser.add_argument("--seed", '-s', type=int, default=1, help="random number seed and data fold")
    parser.add_argument("--time", '-t', required=True, type=str, help="the current time")
    args = vars(parser.parse_args())
    return args


def get_config(config_path):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_result_dir(result_dir, cfg):
    result_dir = osp.join(result_dir, osp.splitext(osp.basename(cfg['config']))[0], cfg['time'].replace(':', '-'), 'train-data_split_seed_' + str(cfg['seed']))
    mkdir_if_missing(result_dir)
    return result_dir


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))


def set_config(config, cfg, save_result_dir):
    config['path_split'] = config['dataset_root_dir'] + config['path_split']
    config['path_feat'] = config['dataset_root_dir'] + config['path_feat']
    config['path_table'] = config['dataset_root_dir'] + config['path_table']

    config['config_file'] = cfg['config']
    config['seed'] = cfg['seed']
    config['time'] = cfg['time']

    config['data_split_seed'] = cfg['seed']
    config['save_result_dir'] = save_result_dir


def init(cfg, config):
    save_result_dir = create_result_dir(config['result_dir'], cfg)
    setup_logger(save_result_dir)
    set_random_seed(cfg['seed'])
    set_config(config, cfg, save_result_dir)


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in config:
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def get_device(gpu_id):
    return torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def get_loss_function(loss_function_name):
    if loss_function_name == 'cross_entropy':
        return F.cross_entropy
    else:
        raise NotImplementedError("Please specify a valid loss function.")


def get_init_key_frequency(config):
    key_frequency = {}
    for dataset_name in config['dataset_names']:
        key_frequency[dataset_name] = []
    return key_frequency


def get_current_ensemble_classes(config, current_dataset):
    class_ensemble_path = config['class_ensemble_path']
    with open(class_ensemble_path) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']

    current_ensemble_classes = {'ensemble_classes': [], 'count': []}
    for k, v in classnames.items():
        for key, val in v.items():
            current_ensemble_classes['count'].append(len(val) * len(templates))
            for name in val:
                current_ensemble_classes['ensemble_classes'].extend([template.replace('CLASSNAME', name) for template in templates])

        if k == current_dataset:
            break

    return current_ensemble_classes


def rename_keys(d, prefix_name, sep='/'):
    newd = dict()
    for k, v in d.items():
        newd[prefix_name + sep + k] = v
    return newd


def check_feature_distribution(feature, label, save_dir, type='image', epoch=0):
    if type == 'text':
        label = torch.arange(feature.shape[1]).unsqueeze(0).repeat(feature.shape[0], 1).view(-1)
        feature = feature.view(-1, feature.shape[-1])

    print('[epoch {}] check {} feature distribution'.format(epoch, type))

    # NaN
    has_nan = torch.isnan(feature).any()
    if has_nan:
        print("[error] Contains NaN")

    # Inf
    has_inf = torch.isinf(feature).any()
    if has_inf:
        print("[error] Contains Inf")

    # tSNE embeddings
    tsne = TSNE()
    X_embedded = tsne.fit_transform(feature.numpy())

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    # Define color palette
    NUM_CLUSTER = torch.max(label).item() + 1
    sns.set(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    palette = sns.color_palette("bright", NUM_CLUSTER)

    # Define custom color mapping for labels
    label_color_mapping = {}
    for i in range(NUM_CLUSTER):
        label_color_mapping[i] = palette[i]

    # Plotting
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=label.numpy(), legend=False,
                    palette=label_color_mapping, ax=ax, s=3, edgecolors='none', linewidths=0)

    plt.savefig(osp.join(save_dir, '{}_{}.png'.format(type, epoch)), bbox_inches='tight')
    wandb.log({type+'_feature': wandb.Image(plt)})
    plt.close()


def plot_key_matching_heatmap(value, path, dataset='train'):
    value = torch.stack(value, dim=0).numpy()
    plt.figure(dpi=500)
    plt.imshow(value, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(osp.join(path, dataset+'_heatmap.png'))
    wandb.log({dataset+'/heatmap': wandb.Image(plt)})
    plt.close()


def get_current_eval_dataloader(data_loader_dict, current_dataset):
    current_eval_dataloader = {}

    for key, val in data_loader_dict.items():
        current_eval_dataloader[key + '/val'] = val['val']
        current_eval_dataloader[key + '/test'] = val['test']
        if key == current_dataset:
            break

    return current_eval_dataloader