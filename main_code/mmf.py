import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import yaml
import argparse
import os
import pickle
from tqdm import tqdm
from transformers import *
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)
import numpy as np
import pdb
import transformers
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
print(transformers.__version__)
def _get_mask(nums, max_num):

    batch_size = nums.size(0)
    max_nums = np.array(nums.cpu()).max()
    arange = torch.arange(0,max_nums).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)

    return non_pad_mask


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        print(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(
        self,
        title_emb,
        video_emb, 
        struct_emb,
        title_mask, 
        video_mask, 
        struct_mask,
        caption_emb = None,
        caption_mask = None,
    ):
        if caption_emb is not None:
            encoder_inputs = torch.cat([video_emb ,title_emb, struct_emb,caption_emb], dim=1)
            attention_mask = torch.cat([video_mask,title_mask,struct_mask,caption_mask], dim=1)

        else:
            encoder_inputs = torch.cat([video_emb ,title_emb, struct_emb], dim=1)
            attention_mask = torch.cat([video_mask,title_mask,struct_mask], dim=1)

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )


        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        
        encoder_outputs = self.encoder(encoder_inputs, extended_attention_mask, head_mask=head_mask)

        
        mmt_seq_output = encoder_outputs[0]

        return mmt_seq_output

class Encoder(nn.Module):
    def __init__(self, path):

        super().__init__()
        f = open(path, 'r', encoding='utf-8')
        cfg = f.read()
        self.config = yaml.load(cfg,Loader=yaml.FullLoader)['model_config']['encoder']
        self.mmt_config = BertConfig(**self.config['mmf'])
        self.mmt = MMT(self.mmt_config)
        self.text_encoder = nn.Embedding(30000,768)
        self.video_encoder = Conv1D(nf=768, nx=2048)
        self.act = ACT2FN["gelu_new"]

    def forward(self,video_feature,frames,title_id,len_title,struct_words_id,len_struct_word,caption_id=None,len_caption=None):

        fwd_results = {}

        self._forward_video_encoding(video_feature,frames,fwd_results)

        self._forward_struct_word_encoding(struct_words_id,len_struct_word,fwd_results)
        self._forward_title_encoding(title_id,len_title,fwd_results)
        if caption_id is not None:
            self._forward_caption_encoding(caption_id,len_caption,fwd_results)
        else:
            fwd_results["caption_embed"] = None
            fwd_results["caption_mask"] = None

        transformer_embedding = self._forward_mmt(fwd_results)

        return transformer_embedding
    
    def _forward_title_encoding(self,titles,title_nums, fwd_results):
        fwd_results["title_embed"] = self.text_encoder(titles)
        max_nums = title_nums[0]
        fwd_results["title_mask"] = _get_mask(
            title_nums,max_nums
        ).cuda()
    
    def _forward_struct_word_encoding(self, struct_words, struct_word_nums, fwd_results):
        fwd_results["struct_word_embed"] = self.text_encoder(struct_words)
        max_nums = struct_word_nums[0]
        fwd_results["struct_word_mask"] = _get_mask(
            struct_word_nums,max_nums
        ).cuda()
    
    def _forward_video_encoding(self,videos,frame_nums,fwd_results):
        fwd_results["video_embed"]= self.act(self.video_encoder(videos))
        max_nums = frame_nums[0]
        fwd_results["video_mask"] = _get_mask(frame_nums, max_nums).cuda()
    

    def _forward_caption_encoding(self, caption, caption_nums, fwd_results):
        fwd_results["caption_embed"] = self.text_encoder(caption)
        max_nums = caption_nums[0]
        fwd_results["caption_mask"] = _get_mask(
            caption_nums,max_nums
        ).cuda()


    def _forward_mmt(self,fwd_results):
        # first forward the text BERT layers
        mmt_results = self.mmt(
            title_emb=fwd_results["title_embed"],
            video_emb=fwd_results["video_embed"], 
            struct_emb=fwd_results["struct_word_embed"],
            caption_emb = fwd_results["caption_embed"],

            title_mask=fwd_results["title_mask"], 
            video_mask=fwd_results["video_mask"], 
            struct_mask=fwd_results["struct_word_mask"],
            caption_mask=fwd_results["caption_mask"],
        )

        return mmt_results

