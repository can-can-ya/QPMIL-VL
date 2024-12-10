import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataloader import default_collate
import os.path as osp
import h5py
from torch import Tensor

from .WSI import WSIClf


def read_datasplit_npz(path: str):
    data_npz = np.load(path, allow_pickle=True)

    pids_train = [str(s) for s in data_npz['train_patients']]
    if 'val_patients' in data_npz:
        pids_val = [str(s) for s in data_npz['val_patients']]
    else:
        pids_val = None
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_data_loaders(cfg):
    print('[infor] Loading data...')
    data_loaders = {}

    for dataset_name in cfg['dataset_names']:
        print('*' * 10, dataset_name.upper(), '*' * 10)
        data_loader = {}

        # Prepare data splitting
        path_split = cfg['path_split'].format(dataset_name, cfg['data_split_seed'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        print('pids_train: count:', len(pids_train), 'value:', pids_train)
        print('pids_val: count:', len(pids_val), 'value:', pids_val)
        print('pids_test: count:', len(pids_test), 'value:', pids_test)
        print('pids: total:', len(pids_train) + len(pids_val) + len(pids_test))

        # Prepare dataset and dataloader
        if cfg['base_model_arch'] == 'CONCH':
            path_feat = cfg['path_feat'].format(dataset_name, cfg['conch_path_feat'])
        else:
            raise NotImplementedError("Please specify a valid architecture.")
        path_table = cfg['path_table'].format(dataset_name, dataset_name.upper())

        train_set = WSIClf(pids_train, path_feat, path_table, cfg['feat_format'])
        print('sids_train: count:', len(train_set), 'value:', train_set.sids)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'],
                                  generator=seed_generator(cfg['data_split_seed']), num_workers=cfg['num_workers'],
                                  shuffle=True, worker_init_fn=seed_worker, collate_fn=default_collate
                                  )
        data_loader['train'] = train_loader

        val_set = WSIClf(pids_val, path_feat, path_table, cfg['feat_format'])
        print('sids_val: count:', len(val_set), 'value:', val_set.sids)
        val_loader = DataLoader(val_set, batch_size=cfg['batch_size'],
                                  generator=seed_generator(cfg['data_split_seed']), num_workers=cfg['num_workers'],
                                  shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
                                  )
        data_loader['val'] = val_loader

        test_set = WSIClf(pids_test, path_feat, path_table, cfg['feat_format'])
        print('sids_test: count:', len(test_set), 'value:', test_set.sids)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'],
                                  generator=seed_generator(cfg['data_split_seed']), num_workers=cfg['num_workers'],
                                  shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
                                  )
        data_loader['test'] = test_loader

        print('sids: total:', len(train_set) + len(val_set) + len(test_set))

        pids_train_label_count = {}
        for label in train_set.pid2label.values():
            if label not in pids_train_label_count:
                pids_train_label_count[label] = 1
            else:
                pids_train_label_count[label] += 1
        print('pids_train_label_count:', pids_train_label_count)
        pids_val_label_count = {}
        for label in val_set.pid2label.values():
            if label not in pids_val_label_count:
                pids_val_label_count[label] = 1
            else:
                pids_val_label_count[label] += 1
        print('pids_val_label_count:', pids_val_label_count)
        pids_test_label_count = {}
        for label in test_set.pid2label.values():
            if label not in pids_test_label_count:
                pids_test_label_count[label] = 1
            else:
                pids_test_label_count[label] += 1
        print('pids_test_label_count:', pids_test_label_count)
        sids_train_label_count = {}
        for label in train_set.sid2label.values():
            if label not in sids_train_label_count:
                sids_train_label_count[label] = 1
            else:
                sids_train_label_count[label] += 1
        print('sids_train_label_count:', sids_train_label_count)
        sids_val_label_count = {}
        for label in val_set.sid2label.values():
            if label not in sids_val_label_count:
                sids_val_label_count[label] = 1
            else:
                sids_val_label_count[label] += 1
        print('sids_val_label_count:', sids_val_label_count)
        sids_test_label_count = {}
        for label in test_set.sid2label.values():
            if label not in sids_test_label_count:
                sids_test_label_count[label] = 1
            else:
                sids_test_label_count[label] += 1
        print('sids_test_label_count:', sids_test_label_count)

        data_loaders[dataset_name] = data_loader

    return data_loaders


def retrieve_from_table_clf(patient_ids, table_path, ret=None, level='slide', shuffle=False,
    processing_table=None, pid_column='patient_id'):
    """Get info from table, oriented to classification tasks"""
    assert level in ['slide', 'patient']
    if ret is None:
        if level == 'patient':
            ret = ['pid', 'pid2sid', 'pid2label']  # for patient-level task
        else:
            ret = ['sid', 'sid2pid', 'sid2label']  # for slide-level task
    for r in ret:
        assert r in ['pid', 'sid', 'pid2sid', 'sid2pid', 'pid2label', 'sid2label']

    df = pd.read_csv(table_path, dtype={pid_column: str})
    assert_columns = [pid_column, 'pathology_id', 'label']
    for c in assert_columns:
        assert c in df.columns
    if processing_table is not None and callable(processing_table):
        df = processing_table(df)

    pid2loc = dict()
    for i in df.index:
        _p = df.loc[i, pid_column]
        if _p in patient_ids:
            if _p in pid2loc:
                pid2loc[_p].append(i)
            else:
                pid2loc[_p] = [i]

    pid, sid = list(), list()
    pid2sid, pid2label, sid2pid, sid2label = dict(), dict(), dict(), dict()
    for p in patient_ids:
        if p not in pid2loc:
            print('[Warning] Patient ID {} is not found in table {}.'.format(p, table_path))
            continue
        pid.append(p)
        for _i in pid2loc[p]:
            _pid, _sid, _label = df.loc[_i, assert_columns].to_list()
            if _pid in pid2sid:
                pid2sid[_pid].append(_sid)
            else:
                pid2sid[_pid] = [_sid]
            if _pid not in pid2label:
                pid2label[_pid] = _label

            sid.append(_sid)
            sid2pid[_sid] = _pid
            sid2label[_sid] = _label

    if shuffle:
        if level == 'patient':
            pid = random.shuffle(pid)
        else:
            sid = random.shuffle(sid)

    res = []
    for r in ret:
        res.append(eval(r))
    return res


def read_patch_data(path: str, dtype: str='torch', key='features'):
    r"""Read patch data from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
        key (string): Key of return data, default 'features'.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            pdata = hf[key][:]
    elif ext == '.pt':
        pdata = torch.load(path, map_location=torch.device('cpu'))
    elif ext == '.npy':
        pdata = np.load(path)
    else:
        raise ValueError(f'Not support {ext}')

    if isinstance(pdata, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(pdata)
    elif isinstance(pdata, Tensor) and dtype == 'numpy':
        return pdata.numpy()
    else:
        return pdata


def get_sids(data_loader):
    sids = {}
    for key, value in data_loader.items():
        sids[key] = {}
        for k, v in value.items():
            sids[key][k] = v.dataset.sids
    return sids