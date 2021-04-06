import torch
import torch.nn as nn

from utils import l2norm
from model.mlp import MLP

  
class VisualEncoding(nn.Module):
    def __init__(self, in_size, h_sizes, out_size, vnorm=True, drop_p=.5, 
                 mapping_h_sizes=[1024,512], mapping_in_drop_p=.5, mapping_h_drop_ps=[.5, .5],
                 have_last_bn=False, pretrained_model_path=''):
        super(VisualEncoding, self).__init__()
        self.norm = norm

        self.visual_mapping = MLP(in_size=in_size, 
                                  h_sizes=h_sizes, 
                                  out_size=out_size, 
                                  in_drop_p=in_dro_p,
                                  drop_ps=drop_ps,
                                  have_last_bn=have_last_bn)
        
        if pretrained_model_path != '':
            checkpoint = torch.load(pretrained_model_path)
            self.load_state_dict(checkpoint['visual_model'])
    
    def forward(self, v_feats):
        features = self.visual_mapping(v_feats)
        
        if self.norm:
            features = l2norm(features)
        
        return features