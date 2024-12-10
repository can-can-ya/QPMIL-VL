from torch.utils.data import Dataset
import torch
import os.path as osp


class WSIClf(Dataset):
    def __init__(self, patient_ids: list, feat_path: str, table_path: str, feat_format: str):
        super(WSIClf, self).__init__()

        self.read_path = feat_path
        self.read_format = feat_format

        info = ['sid', 'sid2pid', 'sid2label', 'pid2label']
        from .data_utils import retrieve_from_table_clf
        self.sids, self.sid2pid, self.sid2label, self.pid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid = self.sids[index]
        pid = self.sid2pid[sid]
        label = self.sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        full_path = osp.join(self.read_path, sid + '.' + self.read_format)
        from .data_utils import read_patch_data
        feats = read_patch_data(full_path, dtype='torch').to(torch.float)

        return index, feats, label