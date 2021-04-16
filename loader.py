import h5py

import torch
from torch.utils.data import Dataset, DataLoader


class DenseCaptioningDataset(Dataset):
    def __init__(self, h5_dataset, vidxs, cidxs, intervals, caps_count, captions, pos, upos, cap_lens, progs, prog_lens):
        super(DenseCaptioningDataset, self).__init__()

        self.cnn_feats = h5_dataset['cnn_features']
        self.c3d_feats = h5_dataset['c3d_features']
        self.feat_count = h5_dataset['count_features']
        self.frame_tstamps = h5_dataset['frames_tstamp']
        
        # self.cnn_feats = torch.Tensor(17955, 20, 2048)
        # self.c3d_feats = torch.Tensor(17955, 20, 4096)
        # self.feat_count = [20 for _ in range(self.cnn_feats.shape[0])]

        self.vidxs = vidxs
        self.cidxs = cidxs
        self.intervals = intervals
        self.caps_count = caps_count
        self.captions = captions
        self.pos = pos
        self.upos = upos
        self.cap_lens = cap_lens
        self.progs = progs
        self.prog_lens = prog_lens

        print(len(self.feat_count), len(vidxs), len(cidxs), len(intervals), len(caps_count), len(captions), len(pos), len(upos), len(cap_lens), len(progs), len(prog_lens))

    def close_file(self):
        self.h5.close()

    def __getitem__(self, index):
        vidx = self.vidxs[index]
        return vidx, self.cidxs[vidx], self.cnn_feats[vidx], self.c3d_feats[vidx], self.feat_count[vidx], self.frame_tstamps[vidx], self.intervals[vidx], self.caps_count[vidx], self.captions[vidx], self.pos[vidx], self.upos[vidx], self.cap_lens[vidx], self.progs[vidx], self.prog_lens[vidx]

    def __len__(self):
        return len(self.vidxs)


def get_dense_loader(h5_dataset, vidxs, cidxs, intervals, caps_count, captions, pos, upos, cap_lens, progs, prog_lens, batch_size, train=True):
    dataset = DenseCaptioningDataset(h5_dataset, vidxs, cidxs, intervals, caps_count, captions, pos, upos, cap_lens, progs, prog_lens)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)