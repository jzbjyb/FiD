# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import inspect
import time
import functools
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.models.t5.modeling_t5 import T5Attention

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

  if not self.training:
    if hasattr(self, 'collect_for_retrieval'):
      # collect things used for retrieval
      self.collect_for_retrieval(scores, hidden_states)
    if hasattr(self, 'collect_cross_attention'):
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

T5Attention.forward = t5attention_forward

class FiDT5Config(transformers.T5Config):
  def __init__(self,
               *args,
               **kwargs):
    if 'n_layer_two_tower' in kwargs:
      self.n_layer_two_tower = kwargs['n_layer_two_tower']
      del kwargs['n_layer_two_tower']
    super().__init__(*args, **kwargs)

class FiDT5(transformers.T5ForConditionalGeneration):
    config_class = FiDT5Config

    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    @classmethod
    def from_t5(cls, model_name: str, n_layer_two_tower: int = 0):
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        config = cls.config_class(n_layer_two_tower=n_layer_two_tower, **t5.config.to_dict())
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
        self.reset_attention_separate_mask()  # always reset separate mask to clean it up
        return result

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, attention_separate_mask=None, max_length=None):
        self.set_attention_separate_mask(attention_separate_mask)  # set separate mask for encoder
        self.encoder.n_passages = input_ids.size(1)
        result = super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )
        self.reset_attention_separate_mask()  # always reset separate mask to clean it up
        return result

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, n_layer_two_tower=self.config.n_layer_two_tower)

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
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def set_attention_separate_mask(self, attention_separate_mask):
        self.encoder.attention_separate_mask = attention_separate_mask

    def reset_attention_separate_mask(self):
        self.encoder.attention_separate_mask = None

    def get_collected_for_retrieval(self):
        return self.encoder.get_collected_for_retrieval()

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        import transformers
        use_old = transformers.__version__ == '3.0.2'
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            if use_old:
                attn.forward = types.MethodType(cross_attention_forward, attn)
            else:
                attn.collect_cross_attention = True

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self,
                 encoder,
                 use_checkpoint: bool = False,
                 n_layer_two_tower: int = 0):
        super().__init__()

        self.encoder = encoder
        self.use_checkpoint = use_checkpoint
        self.n_layer_two_tower = n_layer_two_tower
        self.apply_t5block_wrapper()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if 'direct' in kwargs and kwargs['direct']:  # no reshaping, used in retrieval
          del kwargs['direct']
          return self.encoder(input_ids, attention_mask, **kwargs)
        n_passages = kwargs['n_passages'] if 'n_passages' in kwargs else self.n_passages
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // n_passages
        input_ids = input_ids.view(bsz*n_passages, passage_length)
        if self.n_layer_two_tower > 0:  # use separate attention mask
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

    def apply_t5block_wrapper(self):
      layers = []
      for i, layer in enumerate(self.encoder.block):
        used_for_retreival = i == self.n_layer_two_tower
        wrapped_layer = T5blockWrapper(
          layer,
          use_checkpoint=self.use_checkpoint,
          use_full_attention=i >= self.n_layer_two_tower,
          used_for_retreival=used_for_retreival)
        if used_for_retreival:
          self.retrieval_t5block = wrapped_layer
        layers.append(wrapped_layer)
      self.encoder.block = nn.ModuleList(layers)

    def get_collected_for_retrieval(self):
      return self.retrieval_t5block.get_collected_for_retrieval()

class T5blockWrapper(torch.nn.Module):
    """
    (1) replacing None outputs by empty tensors, which allows the use of checkpointing.
    (2) added code to handle separate/full attention at different layers.
    """
    def __init__(self,
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

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        # register callback for retrieval
        if self.used_for_retreival and not self.training:
            reg_point = self.module.layer[0].SelfAttention
            reg_point.collect_for_retrieval = types.MethodType(
              functools.partial(
                collect_for_retrieval,
                attention_mask=attention_mask[:, 0].eq(0),
                aggregation_method='sum-max',
                field='query',
                use_hidden_states=True,
              ), reg_point)
        # handle separate/full attention
        if self.use_full_attention:
            # modify (potentially separate) attention mask to make it a full attention
            attention_mask = attention_mask.max(2, keepdim=True)[0]  # (bs, num_heads, 1, seq_len)
        # handle checkpointing
        if self.use_checkpoint and self.training:
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

def collect_for_retrieval(
     self,
     scores: torch.FloatTensor,  # (bs, n_heads, seq_len, seq_len)
     hidden_states: torch.FloatTensor,  # (bs, seq_len, emb_size)
     attention_mask: torch.BoolTensor,  # (bs, seq_len, seq_len)
     aggregation_method: str,
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
  cross_mask = ~attention_mask & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (bs, seq_len, seq_len)
  final = ((scores - (~cross_mask.unsqueeze(1) * 1e5)).max(-1)[0].sum(1) * field_mask).sum(-1) / field_mask.sum(-1)  # (bs)
  self.retrieval = {'two_tower_attn_score': final}

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
