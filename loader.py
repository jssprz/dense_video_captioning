import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class DenseCaptioningDataset(Dataset):
    def __init__(self, h5_file_path, h5_file_group_name, vidxs, vidxs_blcklist, cidxs, intervals, caps_count, captions, caps_sem_enc, pos, upos, cap_lens, progs, prog_lens):
        super(DenseCaptioningDataset, self).__init__()

        # self.cnn_feats = h5_dataset['cnn_features']
        # self.c3d_feats = h5_dataset['c3d_features']
        # self.feat_count = h5_dataset['count_features'][...]
        # self.frame_tstamps = h5_dataset['frames_tstamp'][...]
        
        # self.cnn_feats = torch.Tensor(17955, 20, 2048)
        # self.c3d_feats = torch.Tensor(17955, 20, 4096)
        # self.feat_count = [20 for _ in range(self.cnn_feats.shape[0])]

        self.h5_file_path = h5_file_path
        self.h5_file_group_name = h5_file_group_name
        self.h5_dataset = None

        self.vidxs = vidxs
        self.vidxs_blcklist = vidxs_blcklist
        self.cidxs = cidxs
        self.intervals = intervals
        self.caps_count = caps_count
        self.captions = captions
        self.caps_sem_enc = caps_sem_enc
        self.pos = pos
        self.upos = upos
        self.cap_lens = cap_lens
        self.progs = progs
        self.prog_lens = prog_lens

    def close_h5_file(self):
        self.h5_dataset.close()

    def __getitem__(self, index):
        if self.h5_dataset is None:
            self.h5 = h5py.File(self.h5_file_path, 'r')
            self.h5_dataset = self.h5[self.h5_file_group_name]
            
            self.cnn_feats = self.h5_dataset['cnn_features']
            self.c3d_feats = self.h5_dataset['c3d_features']
            self.feat_count = self.h5_dataset['count_features'][...]
            self.frame_tstamps = self.h5_dataset['frames_tstamp'][...]

        vidx = self.vidxs[index]
        if vidx not in self.vidxs_blcklist:
            return vidx, self.cidxs[vidx], self.cnn_feats[vidx], self.c3d_feats[vidx], self.feat_count[vidx], self.frame_tstamps[vidx], self.intervals[vidx], self.caps_count[vidx], self.captions[vidx], self.caps_sem_enc[vidx], self.pos[vidx], self.upos[vidx], self.cap_lens[vidx], self.progs[vidx], self.prog_lens[vidx]
        return None

    def __len__(self):
        return len(self.vidxs)


def filter_blacklist_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x:x is not None, batch))
    return default_collate(batch)


def extract_split_data_from_corpus(corpus, split=0):
    split_data = corpus[split]

    # parse split data from corpus
    vidxs = split_data[0]
    cidxs = split_data[1]
    intervals = split_data[2]
    fps = split_data[3]
    progs = split_data[4]
    prog_lens = [len(p) for p in split_data[4]]
    caps = split_data[5]
    pos = split_data[6] 
    upos = split_data[7]
    cap_lens = [[len(c) for c in caps] for caps in split_data[6]]

    return vidxs, cidxs, intervals, fps, progs, prog_lens, caps, pos, upos, cap_lens


def data2tensors(cidxs, intervals, progs, prog_lens, caps, pos, upos, cap_lens, max_prog=None, max_caps=None, max_words=None):
    if max_prog is None:
        max_prog = max(prog_lens)
    if max_words is None:
        max_words = max([l for v_lens in cap_lens for l in v_lens])
    caps_count_t = torch.tensor([len(v_caps) for v_caps in caps], dtype=torch.int8)
    if max_caps is None:
        max_caps = torch.max(caps_count_t)

    caps_t = torch.LongTensor(len(caps), max_caps, max_words).fill_(0)
    cap_lens_t = torch.LongTensor(len(caps), max_caps).fill_(0)
    cidxs_t = torch.LongTensor(len(caps), max_caps).fill_(0)
    for i, (v_cidxs, v_caps) in enumerate(zip(cidxs, caps)):
        v_caps_t = torch.LongTensor(max_caps, max_words).fill_(0)
        cidxs_t[i, :len(v_cidxs)] = torch.tensor(v_cidxs)
        for j, (cidx, c) in enumerate(zip(v_cidxs, v_caps)):
            v_caps_t[j, :len(c)] = torch.tensor(c[:max_words])
            cap_lens_t[i, j] = len(c)
        caps_t[i] = v_caps_t

    pos_t = torch.zeros((len(caps), max_caps, max_words), dtype=torch.long)
    for i, v_pos in enumerate(pos):
        v_pos_t = torch.zeros((max_caps, max_words), dtype=torch.int8)
        for j, c in enumerate(v_pos):
            v_pos_t[j, :len(c)] = torch.tensor(c[:max_words])
        pos_t[i] = v_pos_t

    upos_t = torch.zeros((len(caps), max_caps, max_words), dtype=torch.long)
    for i, v_upos in enumerate(upos):
        v_upos_t = torch.zeros((max_caps, max_words), dtype=torch.int8)
        for j, c in enumerate(v_upos):
            v_upos_t[j, :len(c)] = torch.tensor(c[:max_words])
        upos_t[i] = v_upos_t

    intervals_t = torch.zeros((len(caps), max_caps, 2))
    for i, v_intervals in enumerate(intervals):
        intervals_t[i, :len(v_intervals)] = torch.Tensor([[s,e] for s, e in v_intervals])

    progs_t = torch.zeros((len(caps), max_prog), dtype=torch.long)
    for i, v_prog in enumerate(progs):
        progs_t[i, :len(v_prog)] = torch.LongTensor(v_prog)

    return cidxs_t, intervals_t, caps_count_t, progs_t, caps_t, pos_t, upos_t, cap_lens_t


def get_dense_loader(h5_file_path, h5_file_group_name, vidxs, vidxs_blcklist, cidxs, intervals, caps_count, captions, caps_sem_enc, pos, upos, cap_lens, progs, prog_lens, batch_size, train=True, num_workers=4, pin_memory=True):
    dataset = DenseCaptioningDataset(h5_file_path, h5_file_group_name, vidxs, vidxs_blcklist, cidxs, intervals, caps_count, captions, caps_sem_enc, pos, upos, cap_lens, progs, prog_lens)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, collate_fn=filter_blacklist_collate, num_workers=num_workers, pin_memory=pin_memory)