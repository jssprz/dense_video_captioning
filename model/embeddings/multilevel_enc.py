import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import l2norm
from model.mlp import MLP


class VisualMultiLevelEncoding(nn.Module):
    def __init__(self, cnn_feats_size, c3d_feats_size, global_feat_size, out_size, 
                 norm=True, rnn_size=1024, drop_p=.5, mapping_h_sizes=[1024,512], mapping_in_drop_p=.5, mapping_h_drop_ps=[.5, .5], 
                 concate=['bow', 'gru', 'cnn'], cnn_out_channels=8, cnn_kernel_sizes=[2,3,4], have_last_bn=False, pretrained_model_path=''):
        super(VisualMultiLevelEncoding, self).__init__()
        self.norm = norm
        
        pool_channels = 512
        self.cnn_conv = nn.Conv1d(in_channels=1, out_channels=pool_channels,
                                  kernel_size=3*cnn_feats_size, stride=cnn_feats_size)
        
        self.c3d_conv = nn.Conv1d(in_channels=1, out_channels=pool_channels,
                                  kernel_size=3*c3d_feats_size, stride=c3d_feats_size)
        
        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(cnn_feats_size, rnn_size, batch_first=True, bidirectional=True)
        # self.rnn = nn.LSTM(word_dim, rnn_size, batch_first=True, bidirectional=True)
        # self.gru_drop = nn.Dropout(p=drop_p)
        self.rnn_output_size = rnn_size*2

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, cnn_out_channels, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in cnn_kernel_sizes
            ])
        
        mapping_in_size = 0
        self.concat_bow = 'bow' in concate
        self.concat_rnn = 'gru' in concate
        self.concat_cnn = 'cnn' in concate
        if self.concat_bow:
            mapping_in_size += pool_channels*2 + global_feat_size
        if self.concat_rnn:
            mapping_in_size += self.rnn_output_size
        if self.concat_cnn:
            mapping_in_size += cnn_out_channels * len(cnn_kernel_sizes) 
        
        # multi fc layers
        self.visual_mapping = MLP(in_size=mapping_in_size, h_sizes=mapping_h_sizes, 
                                  out_size=out_size, in_drop_p=mapping_in_drop_p,
                                  drop_ps=mapping_h_drop_ps, have_last_bn=have_last_bn)

        if pretrained_model_path != '':
            checkpoint = torch.load(pretrained_model_path)
            self.load_state_dict(checkpoint['visual_model'])
        else:
            self.init_weights()
            
    def init_weights(self):
        pass
    
    def freeze_model(self):
        for name, p in model.named_parameters():
            if 'visual_mapping' not in name:
                p.requires_grad = False

    def forward(self, cnn_feats, c3d_feats, video_global_feat, lengths=None):
        # Level 1. Global Encoding by Mean Pooling
        cnn_pool = cnn_feats.view(cnn_feats.shape[0], 1, -1)
        cnn_pool = torch.relu(self.cnn_conv(cnn_pool))
        cnn_pool = cnn_pool.mean(dim=2)
        
        c3d_pool = c3d_feats.view(c3d_feats.shape[0], 1, -1)
        c3d_pool = torch.relu(self.c3d_conv(c3d_pool))
        c3d_pool = c3d_pool.mean(dim=2)
        
        org_out = torch.cat((cnn_pool, c3d_pool, video_global_feat), dim=1)

        # Level 2. Temporal-Aware Encoding by biGRU
        rnn_out, rnn_h = self.rnn(cnn_feats)
        # mean_rnn = torch.zeros(rnn_out.size(0), self.rnn_output_size).to(cnn_feats.device)
        # for i, batch in enumerate(rnn_out):
        #     mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        # rnn_out = mean_rnn
        # rnn_out = self.dropout(rnn_out)
        rnn_h = torch.cat([rnn_h[0,:, :], rnn_h[1,:,:]], dim=1)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        conv_out = rnn_out.unsqueeze(1)
        conv_out = [torch.relu(conv(conv_out)).squeeze(3) for conv in self.convs1]
        conv_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_out]
        conv_out = torch.cat(conv_out, 1)
        # conv_out = self.dropout(conv_out)

        # Levels' outputs concatenation
        features = []
        if self.concat_bow:
            features.append(org_out)
        if self.concat_rnn:
            features.append(rnn_h)
        if self.concat_cnn:
            features.append(conv_out)
        features = torch.cat(features, dim=1)
        
        # mapping to common space
        features = self.visual_mapping(features)
        if self.norm:
            features = l2norm(features)

        return features


class TextMultiLevelEncoding(nn.Module):
    def __init__(self, vocab_size, out_size, norm=True, word_dim=468, we_parameter=None,
                 rnn_size=1024, drop_p=.5, mapping_h_sizes=[1024,512], mapping_in_drop_p=.5, mapping_h_drop_ps=[.5, .5],
                 concate=['bow', 'gru', 'cnn'], cnn_out_channels=8, cnn_kernel_sizes=[2,3,4], have_last_bn=False, pretrained_model_path=''):
        super(TextMultiLevelEncoding, self).__init__()
        self.norm = norm
        self.word_dim = word_dim
        self.we_parameter = we_parameter
        
        #embedding layer
        self.embed = nn.Embedding(vocab_size, word_dim)
        
        # 1-d convolutional network
        pool_channels = 512
        self.embed_conv = nn.Conv1d(in_channels=1, out_channels=pool_channels,
                                    kernel_size=3*word_dim, stride=word_dim)
               
        # bidirectional rnn encoder
        self.rnn = nn.GRU(word_dim, rnn_size, batch_first=True, bidirectional=True)
        # self.rnn = nn.LSTM(word_dim, rnn_size, batch_first=True, bidirectional=True)
        # self.gru_drop = nn.Dropout(p=drop_p)
        
        self.rnn_output_size = rnn_size*2

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, cnn_out_channels, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in cnn_kernel_sizes
            ])

        mapping_in_size = 0
        self.concat_bow = 'bow' in concate
        self.concat_pool = 'pool' in concate
        self.concat_rnn = 'gru' in concate
        self.concat_cnn = 'cnn' in concate
        if self.concat_bow:
            mapping_in_size += vocab_size
        if self.concat_pool:
            mapping_in_size += pool_channels
        if self.concat_rnn:
            mapping_in_size += self.rnn_output_size
        if self.concat_cnn:
            mapping_in_size += cnn_out_channels * len(cnn_kernel_sizes)            
        
        # multi fc layers
        # self.text_mapping = MFC(text_mapping_layers, drop_p, have_bn=True, have_last_bn=True)
        self.text_mapping = MLP(in_size=mapping_in_size, h_sizes=mapping_h_sizes, out_size=out_size, in_drop_p=mapping_in_drop_p, drop_ps=mapping_h_drop_ps, have_last_bn=have_last_bn)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, cap_wids, cap_len, cap_bow, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows = x
        # cap_wids, cap_bows, lengths = text

        # Level 1. Global Encoding by Max Pooling According
        embeddings = self.embed(cap_wids)
        
        embed_pool = embeddings.view(embeddings.shape[0], 1, -1)
        embed_pool = torch.relu(self.embed_conv(embed_pool))
        embed_pool = embed_pool.max(dim=2)[0]

        # Level 2. Temporal-Aware Encoding by biGRU
        packed = pack_padded_sequence(embeddings, cap_len, batch_first=True)
        rnn_out, rnn_h = self.rnn(packed)
        # rnn_out, (rnn_h, rnn_c) = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        rnn_out, lens_unpacked = pad_packed_sequence(rnn_out, batch_first=True)
        
        # print(gru_init_out[0].size(), torch.all(gru_init_out[0][lengths[0]:, :].cpu() == torch.zeros(gru_init_out[0].size(0)-lengths[0], 2048)))
        
        # gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).to(cap_bows.device)
        # for i, batch in enumerate(padded[0]):
        #     gru_out[i] = torch.mean(batch[:lengths[i]], 0)

        # gru_out = torch.stack([torch.mean(gru_out[i, :l, :], dim=0) for i, l in enumerate(lengths)])
        # gru_out = self.gru_drop(gru_out)
        rnn_h = torch.cat([rnn_h[0,:, :], rnn_h[1,:,:]], dim=1)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = rnn_out.unsqueeze(1)
        con_out = [torch.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        # con_out = self.dropout(con_out)
        
        # Levels' outputs concatenation
        features = []
        if self.concat_bow:
            features.append(cap_bow)
        if self.concat_pool:
            features.append(embed_pool)
        if self.concat_rnn:
            features.append(rnn_h)
        if self.concat_cnn:
            features.append(con_out)
        features = torch.cat(features, dim=1)

        # if self.concate == 'full': # level 1+2+3
        #     features = torch.cat((gru_out,con_out,org_out), 1)
        # elif self.concate == 'reduced': # level 2+3
        #     features = torch.cat((gru_out,con_out), 1)
        
        # mapping to common space
        features = self.text_mapping(features)
        if self.norm:
            features = l2norm(features)

        return features
