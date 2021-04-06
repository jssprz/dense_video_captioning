import torch
import torch.nn as nn

from model.captioning.sem_syn_an.encoder import SCNEncoder
from model.captioning.sem_syn_an.decoder import SemSynANDecoder


class SemSynAN(nn.Module):
    def __init__(self, ):
        super(SemSynAN, self).__init__()

        self.encoder = SCNEncoder(...)
        self.decoder = SemSynANDecoder(...)

    def forward(self, video_features):
        # get encoder inputs from video_features
        ...
        
        return self.decoder(self.encoder(...))