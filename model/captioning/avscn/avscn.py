import torch
import torch.nn as nn

from model.captioning.avscn.encoder import SCNEncoder
from model.captioning.avscn.decoder import AVSCNDecoder


class AVSCN(nn.Module):
    def __init__(self, ):
        super(SemSynAN, self).__init__()

        self.encoder = SCNEncoder(...)
        self.decoder = AVSCNDecoder(...)

    def forward(self, video_features):
        # get encoder inputs from video_features
        ...
        
        return self.decoder(self.encoder(...))