# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List
import types
import torch
import transformers
import functools
import torch.nn.functional as F
from torch import nn
import numpy as np
import wandb
from transformers.models.t5.modeling_t5 import T5Attention
from .util import max_sparsify, WandbLogger

def t5attention_forward(
     self,
     hidden_states,
     mask=None,
     key_value_states=None,
     position_bias=None,
     past_key_value=None,
     layer_head_mask=None,
     query_length=None,
     use_cache=False,
     output_attentions=False,
):
  """
  Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
  """
  # Input is (batch_size, seq_length, dim)
  # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
  # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
  batch_size, seq_length = hidden_states.shape[:2]

  real_seq_length = seq_length

  if past_key_value is not None:
    assert (
         len(past_key_value) == 2
    ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
    real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

  key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

  def shape(states):
    """projection"""
    return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

  def unshape(states):
    """reshape"""
    return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

  def project(hidden_states, proj_layer, key_value_states, past_key_value):
    """projects hidden states correctly to key/query states"""
    if key_value_states is None:
      # self-attn
      # (batch_size, n_heads, seq_length, dim_per_head)
      hidden_states = shape(proj_layer(hidden_states))
    elif past_key_value is None:
      # cross-attn
      # (batch_size, n_heads, seq_length, dim_per_head)
      hidden_states = shape(proj_layer(key_value_states))

    if past_key_value is not None:
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, key_length, dim_per_head)
        hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
      else:
        # cross-attn
        hidden_states = past_key_value
    return hidden_states

  # get query states
  query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

  # get key/value states
  key_states = project(
    hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
  )
  value_states = project(
    hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
  )

  # compute scores
  scores = torch.matmul(
    query_states, key_states.transpose(3, 2)
  )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

  if position_bias is None:
    if not self.has_relative_attention_bias:
      position_bias = torch.zeros(
        (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
      )
      if self.gradient_checkpointing and self.training:
        position_bias.requires_grad = True
    else:
      position_bias = self.compute_bias(real_seq_length, key_length)

    # if key and values are already calculated
    # we want only the last query position bias
    if past_key_value is not None:
      position_bias = position_bias[:, :, -hidden_states.size(1):, :]

  # TODO: this is the key change! (any better workaround?)
  scores += position_bias
  if mask is not None:
    scores = scores + mask  # (batch_size, n_heads, seq_length, key_length)

  # compute the KL between encoder and decoder attn
  if self.is_decoder and hasattr(self, 'encoder_imp') and hasattr(self, 'encoder_decoder_kl'):
    self.kl_loss = self.encoder_decoder_kl(scores, mask.eq(0), self.encoder_imp)
  # modify decoder attn based on encoder
  if self.is_decoder and hasattr(self, 'encoder_imp') and not hasattr(self, 'encoder_decoder_kl'):
    scores += self.encoder_imp + self.encoder_imp_mask

  attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    scores
  )  # (batch_size, n_heads, seq_length, key_length)
  attn_weights = nn.functional.dropout(
    attn_weights, p=self.dropout, training=self.training
  )  # (batch_size, n_heads, seq_length, key_length)
  # collect encoder attn
  if hasattr(self, 'collect_for_retrieval'):
    self.collect_for_retrieval(scores, hidden_states)
  # collect decoder attn
  if self.is_decoder and not self.training and hasattr(self, 'collect_cross_attention'):
    self.collect_cross_attention(scores, mask.eq(0))

  # Mask heads if we want to
  if layer_head_mask is not None:
    attn_weights = attn_weights * layer_head_mask

  attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
  attn_output = self.o(attn_output)

  present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
  outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

  if output_attentions:
    outputs = outputs + (attn_weights,)
  return outputs

T5Attention.forward = t5attention_forward

class FiDT5Config(transformers.T5Config):
  def __init__(self,
               *args,
               n_layer_two_tower: int = 0,
               attention_mask: str = 'separate',
               query_in_decoder: str = 'no',
               num_keep_ctx_in_decoder: int = 0,
               keep_ctx_in_decoder_with_head: int = None,
               encoder_decoder_kl_ratio: float = 0,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.n_layer_two_tower = n_layer_two_tower
    self.attention_mask = attention_mask
    self.query_in_decoder = query_in_decoder
    self.num_keep_ctx_in_decoder = num_keep_ctx_in_decoder
    self.keep_ctx_in_decoder_with_head = keep_ctx_in_decoder_with_head
    self.encoder_decoder_kl_ratio = encoder_decoder_kl_ratio

class FiDT5(transformers.T5ForConditionalGeneration):
    config_class = FiDT5Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wrap_encoder(config)
        self.wrap_decoder(config)
        self.collect_kl_loss_from_decoder = bool(config.encoder_decoder_kl_ratio)

    @classmethod
    def from_t5(cls,
                model_name: str,
                n_layer_two_tower: int = 0,
                attention_mask: str = 'separate',
                query_in_decoder: str = 'no',
                num_keep_ctx_in_decoder: int = 0,
                keep_ctx_in_decoder_with_head: int = None,
                encoder_decoder_kl_ratio: float = 0):
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        config = cls.config_class(
          n_layer_two_tower=n_layer_two_tower,
          attention_mask=attention_mask,
          query_in_decoder=query_in_decoder,
          num_keep_ctx_in_decoder=num_keep_ctx_in_decoder,
          keep_ctx_in_decoder_with_head=keep_ctx_in_decoder_with_head,
          encoder_decoder_kl_ratio=encoder_decoder_kl_ratio,
          **t5.config.to_dict())
        model = cls(config)
        model.load_t5(t5.state_dict())
        return model

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if 'attention_separate_mask' in kwargs:  # set separate mask for encoder
            asm = kwargs['attention_separate_mask']
            self.set_attention_separate_mask(asm)
            del kwargs['attention_separate_mask']
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        if self.collect_kl_loss_from_decoder:
          kl_loss = self.decoder.get_kl_loss()
          if 'return_dict' in kwargs and kwargs['return_dict'] and result.loss is not None:
            result.loss = result.loss + kl_loss
          elif 'labels' in kwargs and kwargs['labels'] is not None:
            result = (result[0] + kl_loss,) + result[1:]
        self.reset_attention_separate_mask()  # always reset separate mask to clean it up
        return result

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, attention_separate_mask=None, max_length=None, **kwargs):
        self.set_attention_separate_mask(attention_separate_mask)  # set separate mask for encoder
        self.encoder.n_passages = input_ids.size(1)
        result = super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )
        self.reset_attention_separate_mask()  # always reset separate mask to clean it up
        return result

    def wrap_decoder(self, config):
      if not self.need_wrap_decoder(config):
        return
      get_encoder_importance_func = lambda: self.get_collected_for_retrieval()['two_tower_attn_score']
      self.decoder = DecoderWrapper(config, self.decoder, get_encoder_importance_func)

    def unwrap_decoder(self, config):
      if not self.need_wrap_decoder(config):
        return
      self.decoder = self.decoder.decoder

    @staticmethod
    def need_wrap_decoder(config):
      return bool(config.num_keep_ctx_in_decoder) or bool(config.encoder_decoder_kl_ratio)

    def get_inner_decoder(self):
      decoder = self.decoder.decoder if type(self.decoder) is DecoderWrapper else self.decoder
      return decoder

    def wrap_encoder(self, config):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(config, self.encoder)

    def unwrap_encoder(self, keep_checkpoint: bool = False):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        if keep_checkpoint:
            return
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.unwrap_decoder(self.config)
        self.load_state_dict(state_dict)
        self.wrap_encoder(self.config)
        self.wrap_decoder(self.config)

    def load_from(self, model):
      self.encoder.load_state_dict(model.encoder.state_dict())
      if type(self.decoder) == type(model.decoder) == DecoderWrapper:
        self.decoder.load_state_dict(model.decoder.state_dict())
      else:
        self.get_inner_decoder().load_state_dict(model.get_inner_decoder().state_dict())

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def set_attention_separate_mask(self, attention_separate_mask):
        self.encoder.attention_separate_mask = attention_separate_mask

    def reset_attention_separate_mask(self):
        self.encoder.attention_separate_mask = None

    def get_collected_for_retrieval(self):
        return self.encoder.get_collected_for_retrieval()

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.get_inner_decoder().block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(
         self,
         positions: List[int],  # (bs)
         context_mask,  # (bs, n_passages, text_maxlength)
         sum_over_head_and_layer: bool = True):
      """
      Cross-attention scores are aggregated to obtain a single scalar per
      passage. This scalar can be seen as a similarity score between the
      question and the input passage. It is obtained by averaging the
      cross-attention scores obtained on the first decoded token over heads,
      layers, and tokens of the input passage.

      More details in Distilling Knowledge from Reader to Retriever:
      https://arxiv.org/abs/2012.04584.
      """
      n_passages = context_mask.size(1)

      # collect
      scores = []
      for mod in self.get_inner_decoder().block:
        s = torch.cat(mod.layer[1].EncDecAttention.score_storage, dim=2)  # (bs, n_head, decode_len, encode_len)
        scores.append(s)
      scores = torch.stack(scores, dim=2)  # (bs, n_head, n_layers, decode_len, encode_len)
      bsz, n_heads, n_layers, decode_len, _ = scores.size()
      scores = scores.view(bsz, n_heads, n_layers, decode_len, n_passages, -1)  # (bs, n_head, n_layers, decode_len, n_passages, text_maxlength)

      # decide whether score is already summed over text_maxlength
      if scores.size(-1) != context_mask.size(-1) and scores.size(-1) == 1:  # already summed
        context_mask = torch.ones_like(context_mask)[:, :, :1]  # (bs, n_passages, 1)

      # choose the cross attn corresponding to positions
      scores = scores.masked_fill(~context_mask[:, None, None, None], 0.)
      assert len(positions) == bsz
      _scores = []
      for i, p in enumerate(positions):
        _scores.append(scores[i, :, :, p])  # (n_head, n_layers, n_passages, text_maxlength)
      scores = torch.stack(_scores, dim=0)  # (bs, n_head, n_layers, n_passages, text_maxlength)

      # aggregation
      if sum_over_head_and_layer:
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens  # batch_size, n_passages
      else:
        scores = scores.sum(dim=[4])
        ntokens = context_mask.sum(dim=[2])[:, None, None]
        scores = scores / ntokens  # batch_size, n_head, n_layers, n_passages
        scores = scores.permute(0, 3, 2, 1)  # batch_size, n_passages, n_layers, n_head
      return scores

    def overwrite_forward_crossattention(self, n_context: int):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        import transformers
        use_old = transformers.__version__ == '3.0.2'
        for mod in self.get_inner_decoder().block:
            attn = mod.layer[1].EncDecAttention
            if use_old:
                attn.forward = types.MethodType(cross_attention_forward, attn)
            else:
                attn.collect_cross_attention = types.MethodType(
                  functools.partial(collect_cross_attention, n_context=n_context, sum_over_tokens=True), attn)

class DecoderWrapper(torch.nn.Module):
  def __init__(self, config, decoder, get_encoder_importance_func: Callable):
    super().__init__()
    self.decoder = decoder
    self.get_encoder_importance_func = get_encoder_importance_func
    self.num_keep_ctx_in_decoder = config.num_keep_ctx_in_decoder
    self.keep_ctx_in_decoder_with_head = config.keep_ctx_in_decoder_with_head
    self.encoder_decoder_kl_ratio = config.encoder_decoder_kl_ratio
    if self.encoder_decoder_kl_ratio and self.num_keep_ctx_in_decoder:
      raise ValueError('only one of the KL and combined loss should be used')
    if FiDT5.need_wrap_decoder(config):
      # head weight always needed
      if config.keep_ctx_in_decoder_with_head is None:
        self.head_weights = torch.nn.Parameter(torch.zeros(config.num_heads), requires_grad=True)
      else:
        weights = [-1e5] * config.num_heads
        weights[config.keep_ctx_in_decoder_with_head] = 1.0
        self.head_weights = torch.nn.Parameter(torch.tensor(weights), requires_grad=False)
      # combine weight only for combined attn
      if config.num_keep_ctx_in_decoder:
        self.combine_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

  def get_kl_loss(self):
    assert self.encoder_decoder_kl_ratio
    kl_loss = self.decoder.block[-1].layer[1].EncDecAttention.kl_loss * self.encoder_decoder_kl_ratio
    WandbLogger.log_w_step({'kl-loss': kl_loss.item()})
    return kl_loss

  def forward(
       self,
       input_ids=None,
       attention_mask=None,
       encoder_hidden_states=None,
       encoder_attention_mask=None,
       **kwargs):
    # fetch encoder importance
    encoder_imp = self.get_encoder_importance_func()  # (num_q, num_d, num_layer, num_head)
    num_q, num_d, num_layer, num_head = encoder_imp.size()
    num_q, dt, _ = encoder_hidden_states.size()  # (num_q, num_d * ctx_len, emb_size)
    assert dt % num_d == 0, 'encoder_hidden_states shape error'
    ctx_len = dt // num_d
    # use the first layer
    encoder_imp = encoder_imp[:, :, 0, :]  # (num_q, num_d, num_head)
    # combine multiple heads
    encoder_imp = (encoder_imp * torch.softmax(self.head_weights, 0)[None, None, :]).sum(-1)  # (num_q, num_d)
    WandbLogger.log_w_step({'head-weight': torch.softmax(self.head_weights, 0)})
    if self.num_keep_ctx_in_decoder:
      # apply combine weight
      encoder_imp = torch.exp(self.combine_weight) * encoder_imp
      WandbLogger.log_w_step({'combine-weight': torch.exp(self.combine_weight).item()})
      # spasify
      assert self.num_keep_ctx_in_decoder <= num_d
      value, ind = encoder_imp.topk(self.num_keep_ctx_in_decoder, -1)
      encoder_imp_mask = torch.zeros_like(encoder_imp).bool()
      encoder_imp_mask.scatter_(-1, ind, True)  # encoder_imp still contains all values and we use this mask to force sparsity
      # reshape to (num_q, 1 (num_head), 1 (query_len), num_d * ctx_len)
      encoder_imp = encoder_imp.unsqueeze(-1).repeat(1, 1, ctx_len).view(num_q, 1, 1, -1)
      # convert to final mask of shape (num_q, 1 (num_head), 1 (query_len), num_d * ctx_len)
      encoder_imp_mask = self.decoder.invert_attention_mask(encoder_imp_mask.unsqueeze(-1).repeat(1, 1, ctx_len).view(num_q, -1))
      # assign to each cross attn module
      for layer in self.decoder.block:
        layer.layer[1].EncDecAttention.encoder_imp = encoder_imp
        layer.layer[1].EncDecAttention.encoder_imp_mask = encoder_imp_mask
    elif self.encoder_decoder_kl_ratio:
      # assign to the last cross attn module
      reg_point = self.decoder.block[-1].layer[1].EncDecAttention
      reg_point.encoder_imp = encoder_imp
      reg_point.encoder_decoder_kl = types.MethodType(functools.partial(encoder_decoder_kl, n_context=num_d), reg_point)
    else:
      raise NotImplementedError
    # decode
    return self.decoder(input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, **kwargs)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self,
                 config,
                 encoder,
                 use_checkpoint: bool = False):
        super().__init__()

        self.encoder = encoder
        self.use_checkpoint = use_checkpoint
        self.n_layer_two_tower = config.n_layer_two_tower
        self.apply_t5block_wrapper(config)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if 'direct' in kwargs and kwargs['direct']:  # no reshaping, used in retrieval
          del kwargs['direct']
          return self.encoder(input_ids, attention_mask, **kwargs)
        n_passages = kwargs['n_passages'] if 'n_passages' in kwargs else self.n_passages
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // n_passages
        input_ids = input_ids.view(bsz*n_passages, passage_length)
        if hasattr(self, 'attention_separate_mask') and self.attention_separate_mask is not None:  # use separate attention mask
          attention_mask = self.attention_separate_mask.view(
            *((bsz * n_passages,) + self.attention_separate_mask.size()[2:]))  # (bs * n_passage, seq_len * seq_len)
        else:
          attention_mask = attention_mask.view(bsz * n_passages, passage_length)  # (bs * n_passage, seq_len)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        if kwargs['return_dict']:
          outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, n_passages*passage_length, -1)
        else:
          outputs = (outputs[0].view(bsz, n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

    def apply_t5block_wrapper(self, config):
      only_use_first_layer = lambda i: i == config.n_layer_two_tower
      use_all_layers = lambda i: i >= config.n_layer_two_tower
      layers = []
      self.retrieval_t5block_funcs = []
      for i, layer in enumerate(self.encoder.block):
        used_for_retreival = only_use_first_layer(i)
        wrapped_layer = T5blockWrapper(
          config,
          layer,
          use_checkpoint=self.use_checkpoint,
          use_full_attention=i >= config.n_layer_two_tower,
          used_for_retreival=used_for_retreival)
        if used_for_retreival:
          self.retrieval_t5block_funcs.append(wrapped_layer.get_collected_for_retrieval)
        layers.append(wrapped_layer)
      self.encoder.block = nn.ModuleList(layers)

    def get_collected_for_retrieval(self):
      merge = {}
      for i, func in enumerate(self.retrieval_t5block_funcs):
        result = func()
        for k, v in result.items():  # v is (bs, num_heads)
          if k not in merge:
            merge[k] = [v]
          else:
            merge[k].append(v)
      for k in merge:
        merge[k] = torch.stack(merge[k], 1)  # (bs, num_layers, num_heads)
        # (num_query, n_context, num_layers, num_heads)
        merge[k] = merge[k].view(-1, self.n_passages, merge[k].size(1), merge[k].size(2))
      return merge

class T5blockWrapper(torch.nn.Module):
    """
    (1) replacing None outputs by empty tensors, which allows the use of checkpointing.
    (2) added code to handle separate/full attention at different layers.
    """
    def __init__(self,
                 config,
                 module,
                 use_checkpoint: bool = False,
                 use_full_attention: bool = True,
                 used_for_retreival: bool = False):
        assert not used_for_retreival or use_full_attention, 'retrieval layer must use full attention'
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
        self.use_full_attention = use_full_attention
        self.used_for_retreival = used_for_retreival
        self.collect_for_decoder = FiDT5.need_wrap_decoder(config)

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        # register callback for retrieval
        if self.used_for_retreival and (not self.training or self.collect_for_decoder):
            reg_point = self.module.layer[0].SelfAttention
            reg_point.collect_for_retrieval = types.MethodType(
              functools.partial(
                collect_for_retrieval,
                attention_mask=attention_mask[:, 0].eq(0),
                aggregation_method='all-avg-max',
                field='query',
                use_hidden_states=False,
              ), reg_point)
        # handle separate/full attention
        if self.use_full_attention:
            # modify (potentially separate) attention mask to make it a full attention
            attention_mask = attention_mask.max(2, keepdim=True)[0]  # (bs, num_heads, 1, seq_len)
        # handle checkpointing
        if self.use_checkpoint and self.training and not self.used_for_retreival:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

    def get_collected_for_retrieval(self):
        reg_point = self.module.layer[0].SelfAttention
        return reg_point.retrieval

def collect_cross_attention(
     self,
     scores: torch.FloatTensor,  # (bs, n_heads, 1, enc_seq_len)
     mask: torch.BoolTensor,  # (bs, n_heads or 1, 1, enc_seq_len)
     n_context: int,
     sum_over_tokens: bool):
  # save cross attn at all decoding position
  if sum_over_tokens:
    # TODO: we sum over text_len to save space but is there any better workaround?
    s = scores.view(*(scores.size()[:3] + (n_context, -1)))  # (bs, n_heads, 1, n_context, text_len)
    m = mask.view(*(mask.size()[:3] + (n_context, -1)))  # (bs, n_heads or 1, 1, n_context, text_len)
    s = (s * m).sum(-1) / m.sum(-1)  # (bs, n_heads, 1, n_context)
  else:
    s = scores
  if self.score_storage is None:
    self.score_storage = [s]
  else:
    self.score_storage.append(s)

def encoder_decoder_kl(
     self,
     decoder_score: torch.FloatTensor,  # (bs, n_heads, dec_seq_len, enc_seq_len)
     decoder_mask: torch.BoolTensor,  # (bs, n_heads or 1, dec_seq_len, enc_seq_len)
     encoder_score: torch.FloatTensor,  # (bs, n_context)
     n_context: int):
  # avg over doc tokens
  s = decoder_score.view(*(decoder_score.size()[:3] + (n_context, -1)))  # (bs, n_heads, dec_seq_len, n_context, text_len)
  m = decoder_mask.view(*(decoder_mask.size()[:3] + (n_context, -1)))  # (bs, n_heads or 1, dec_seq_len, n_context, text_len)
  s = (s * m).sum(-1) / m.sum(-1)  # (bs, n_heads, dec_seq_len, n_context)
  # avg over head, use the first decoder tok
  s = s.mean(1)[:, 0]  # (bs, n_context)
  # kl
  kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
  dec_attn = torch.softmax(s.detach(), dim=-1)  # no grad to decoder
  enc_attn = torch.nn.functional.log_softmax(encoder_score, dim=-1)
  kl = kl_loss(enc_attn, dec_attn)
  return kl

def collect_for_retrieval(
     self,
     scores: torch.FloatTensor,  # (bs, n_heads, seq_len, seq_len)
     hidden_states: torch.FloatTensor,  # (bs, seq_len, emb_size)
     attention_mask: torch.BoolTensor,  # (bs, seq_len, seq_len)
     aggregation_method: str,  # 'head-query-key'
     field: str,
     use_hidden_states: bool,
):
  if use_hidden_states:
    scores = torch.matmul(
      hidden_states,
      hidden_states.transpose(1, 2)
    )
    scores = scores.unsqueeze(1)

  pad_mask = attention_mask.max(1)[0]  # (bs, seq_len)
  if field == 'query':
    field_mask = attention_mask[:, 0]  # (bs, seq_len)
  elif field == 'doc':
    field_mask = attention_mask[:, -1]  # (bs, seq_len)
  elif field == 'all':
    field_mask = pad_mask
  else:
    raise NotImplementedError
  cross_mask = (~attention_mask & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)).unsqueeze(1)  # (bs, 1, seq_len, seq_len)

  head_agg, query_agg, key_agg = aggregation_method.split('-')
  if key_agg == 'max':
    scores = (scores - (~cross_mask * 1e10)).max(-1)[0]  # (bs, n_heads, seq_len)
  elif key_agg == 'avg':
    scores = (scores * cross_mask).sum(-1) / (cross_mask.sum(-1) + 1e-10)  # (bs, n_heads, seq_len)
  elif key_agg == 'maxsp':
    cross_mask = cross_mask.repeat(1, scores.size(1), 1, 1)
    max_sparsify(scores, cross_mask, -1, inplace=True)  # (bs, n_heads, seq_len, seq_len)
  else:
    raise NotImplementedError

  if query_agg == 'avg':
    scores = (scores * field_mask.unsqueeze(1)).sum(-1) / field_mask.unsqueeze(1).sum(-1)  # (bs, n_heads)
  elif query_agg == 'maxsp':
    assert key_agg == 'maxsp', 'maxsp must be used consecutively'
    max_sparsify(scores, cross_mask, -2, inplace=True)  # (bs, n_heads, seq_len, seq_len)
    scores = (scores.sum(-1) * field_mask[:, None]).sum(-1) / \
             (cross_mask.sum(-1) * field_mask[:, None]).sum(-1)  # (bs, n_heads)
  else:
    raise NotImplementedError

  if head_agg == 'avg':
    scores = scores.mean(-1)  # (bs)
  elif head_agg == 'all':
    pass
  else:
    raise NotImplementedError
  self.retrieval = {'two_tower_attn_score': scores}

def cross_attention_forward_new(
     self,
     hidden_states,
     mask=None,
     key_value_states=None,
     position_bias=None,
     past_key_value=None,
     layer_head_mask=None,
     query_length=None,
     use_cache=False,
     output_attentions=False):
  """
  from: https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/t5/modeling_t5.py
  Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
  """
  # Input is (batch_size, seq_length, dim)
  # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
  # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
  batch_size, seq_length = hidden_states.shape[:2]

  real_seq_length = seq_length

  if past_key_value is not None:
    assert (
         len(past_key_value) == 2
    ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
    real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

  key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

  def shape(states):
    """projection"""
    return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

  def unshape(states):
    """reshape"""
    return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

  def project(hidden_states, proj_layer, key_value_states, past_key_value):
    """projects hidden states correctly to key/query states"""
    if key_value_states is None:
      # self-attn
      # (batch_size, n_heads, seq_length, dim_per_head)
      hidden_states = shape(proj_layer(hidden_states))
    elif past_key_value is None:
      # cross-attn
      # (batch_size, n_heads, seq_length, dim_per_head)
      hidden_states = shape(proj_layer(key_value_states))

    if past_key_value is not None:
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, key_length, dim_per_head)
        hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
      else:
        # cross-attn
        hidden_states = past_key_value
    return hidden_states

  # get query states
  query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

  # get key/value states
  key_states = project(
    hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
  )
  value_states = project(
    hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
  )

  # compute scores
  scores = torch.matmul(
    query_states, key_states.transpose(3, 2)
  )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

  if position_bias is None:
    if not self.has_relative_attention_bias:
      position_bias = torch.zeros(
        (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
      )
      if self.gradient_checkpointing and self.training:
        position_bias.requires_grad = True
    else:
      position_bias = self.compute_bias(real_seq_length, key_length)

    # if key and values are already calculated
    # we want only the last query position bias
    if past_key_value is not None:
      position_bias = position_bias[:, :, -hidden_states.size(1):, :]

    if mask is not None:
      position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

  scores += position_bias
  if self.score_storage is None:
    self.score_storage = scores

  attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    scores
  )  # (batch_size, n_heads, seq_length, key_length)
  attn_weights = nn.functional.dropout(
    attn_weights, p=self.dropout, training=self.training
  )  # (batch_size, n_heads, seq_length, key_length)

  # Mask heads if we want to
  if layer_head_mask is not None:
    attn_weights = attn_weights * layer_head_mask

  attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
  attn_output = self.o(attn_output)

  present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
  outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

  if output_attentions:
    outputs = outputs + (attn_weights,)
  return outputs

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

class RetrieverConfig(transformers.BertConfig):
    def __init__(self,
                 model_name='bert-base-uncased',
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 scale_dot_product=False,
                 **kwargs):
        super().__init__(**transformers.BertConfig.from_pretrained(model_name).to_dict())
        self.model_name = model_name
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls=extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection
        self.scale_dot_product = scale_dot_product

class T5RetrieverConfig(transformers.T5Config):
    def __init__(self,
                 model_name='t5-base',
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 scale_dot_product=False,
                 **kwargs):
        super().__init__(**transformers.T5Config.from_pretrained(model_name).to_dict())
        self.model_name = model_name
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls = extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection
        self.scale_dot_product = scale_dot_product

class RetrieverMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_retriever_class(initialize_with: str):
        if 'reader' in initialize_with or 't5' in initialize_with:
            return T5Retriever
        return Retriever

    def set_checkpoint(self, *args, **kwargs):
        pass

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None):
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        if self.config.scale_dot_product:
            score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.forward_func(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )

        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)


class Retriever(RetrieverMixin, transformers.PreTrainedModel):
  config_class = RetrieverConfig
  base_model_prefix = "retriever"

  def __init__(self, config):
    super().__init__(config)
    self.config = config
    initialize_with = config.model_name
    if initialize_with:
      self.model = transformers.BertModel.from_pretrained(initialize_with)
    else:
      self.model = transformers.BertModel(config)
    if self.config.projection:
      self.proj = nn.Linear(
        self.model.config.hidden_size,
        self.config.indexing_dimension
      )
      self.norm = nn.LayerNorm(self.config.indexing_dimension)
    self.loss_fct = torch.nn.KLDivLoss()
    self.forward_func = self.model.forward

  def load_tokenizer(self):
    return transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')


class T5Retriever(RetrieverMixin, transformers.PreTrainedModel):
  config_class = T5RetrieverConfig
  base_model_prefix = "t5retriever"

  def __init__(self, config):
    super().__init__(config)
    self.config = config
    initialize_with = config.model_name
    if 'reader' in initialize_with:  # FiD
      self.model = FiDT5.from_pretrained(initialize_with)
    elif 't5' in initialize_with:
      t5 = transformers.T5ForConditionalGeneration.from_pretrained(initialize_with)
      self.model = FiDT5(t5.config)
      self.model.load_t5(t5.state_dict())
    else:  # use bert as default
      raise NotImplementedError

    if self.config.projection:
      self.proj = nn.Linear(
        self.model.config.hidden_size,
        self.config.indexing_dimension
      )
      self.norm = nn.LayerNorm(self.config.indexing_dimension)
    self.loss_fct = torch.nn.KLDivLoss()
    self.forward_func = functools.partial(self.model.encoder.forward, direct=True)

  def set_checkpoint(self, use_checkpoint):
    self.model.set_checkpoint(use_checkpoint)

  def load_tokenizer(self):
    return transformers.T5Tokenizer.from_pretrained('t5-base')
