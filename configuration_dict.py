import json
from dataclasses import dataclass


# @dataclass()
# class TaggerConfig:
#     in_size: int
#     h_sizes: list
#     out_size: int
#     in_dropout_p: float
#     dropout_ps: list
#     have_last_bn: bool


# @dataclass
# class EmbeddingConfig:
#     in_size: int
#     h_sizes: list
#     out_size: int
#     norm: bool
#     dropout_p: float
#     have_last_bn: bool
#     pretrained_model_path: str


# @dataclass
# class AVSCNDecoderConfig:
#     in_seq_length: int
#     out_seq_length: int
#     n_feats: int
#     n_tags: int
#     embedding_size: int
#     h_size: int
#     rnn_in_size: int
#     rnn_h_size: int
#     encoder_num_layers: int
#     encoder_bidirectional: bool
#     rnn_cell: str
#     num_layers: int
#     dropout_p: float
#     beam_size: int
#     temperature: float
#     train_sample_max: bool
#     test_sample_max: bool
#     beam_search_logic: str


# @dataclass
# class SemSynANDecoderConfig:
#     in_seq_length: int
#     out_seq_length: int
#     n_feats: int
#     n_tags: int
#     n_pos_emb: int
#     embedding_size: int
#     h_size: int
#     rnn_in_size: int
#     rnn_h_size: int
#     encoder_num_layers: int
#     encoder_bidirectional: bool
#     rnn_cell: str
#     num_layers: int
#     dropout_p: float
#     beam_size: int
#     temperature: float
#     train_sample_max: bool
#     test_sample_max: bool
#     beam_search_logic: str


# @dataclass
# class VNCLCellConfig:
#     m_size: int
#     vh_size: int
#     h1_size: int


def load_config_json(file_path):
    with open(config_path, 'r') as f:
        config = json.load(f)


class ConfigDict:
    def __init__(self, dictionary):
        self.parse_config(dictionary)

    def parse_config(self, dictionary):
        for k, v in dictionary.items():
            if type(v) == dict:
                setattr(self, k, ConfigDict(v))
            else:
                setattr(self, k, v)