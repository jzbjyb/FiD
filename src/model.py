# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cmath import log
from mimetypes import init
from typing import Callable, List, Dict
import types
import warnings
import random
import contextlib
import torch
import transformers
import functools
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from transformers.models.t5.modeling_t5 import \
  T5Attention, T5Stack, T5LayerSelfAttention, T5ForConditionalGeneration, __HEAD_MASK_WARNING_MSG, logger
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqLMOutput
from entmax import sparsemax
from src.util import RandContext, max_sparsify, WandbLogger, global_context
from src.dist_utils import all_gather_tensors, get_rank, get_world_size
from src.index import Indexer
from src.memory_bank import MemoryBank, MemoryBankProcessHelper


def t5forconditionalgeneration_forward(
     self,
     input_ids=None,
     attention_mask=None,
     decoder_input_ids=None,
     decoder_attention_mask=None,
     head_mask=None,
     decoder_head_mask=None,
     cross_attn_head_mask=None,
     encoder_outputs=None,
     past_key_values=None,
     inputs_embeds=None,
     decoder_inputs_embeds=None,
     labels=None,
     use_cache=None,
     output_attentions=None,
     output_hidden_states=None,
     return_dict=None,
     input_doc_ids=None,  # document identifiers 
     gold_doc_dist=None,  # gold document annotation
     only_bi_encoder_forward: bool = False
):
  r"""
  labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
      Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
      labels in `[0, ..., config.vocab_size]`
  Returns:
  Examples:
  ```python
  >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
  >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
  >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
  >>> # training
  >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
  >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
  >>> outputs = model(input_ids=input_ids, labels=labels)
  >>> loss = outputs.loss
  >>> logits = outputs.logits
  >>> # inference
  >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
  >>> outputs = model.generate(input_ids)
  >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  >>> # studies have shown that owning a dog is good for you.
  ```"""
  use_cache = use_cache if use_cache is not None else self.config.use_cache
  return_dict = return_dict if return_dict is not None else self.config.use_return_dict

  # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
  if head_mask is not None and decoder_head_mask is None:
    if self.config.num_layers == self.config.num_decoder_layers:
      warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
      decoder_head_mask = head_mask

  # Encode if needed (training, first prediction pass)
  if encoder_outputs is None:
    # Convert encoder inputs in embeddings if needed
    encoder_outputs = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask,
      inputs_embeds=inputs_embeds,
      head_mask=head_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
      input_doc_ids=input_doc_ids,
      only_bi_encoder_forward=only_bi_encoder_forward)
    if only_bi_encoder_forward:
      return encoder_outputs
  elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
    encoder_outputs = BaseModelOutput(
      last_hidden_state=encoder_outputs[0],
      hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
      attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
    )

  hidden_states = encoder_outputs[0]

  if self.model_parallel:
    torch.cuda.set_device(self.decoder.first_device)

  if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
    # get decoder inputs from shifting lm labels to the right
    decoder_input_ids = self._shift_right(labels)

  # Set device for model parallelism
  if self.model_parallel:
    torch.cuda.set_device(self.decoder.first_device)
    hidden_states = hidden_states.to(self.decoder.first_device)
    if decoder_input_ids is not None:
      decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.decoder.first_device)
    if decoder_attention_mask is not None:
      decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

  # Decode
  if hasattr(self.encoder, 'attention_mask'):  # (bs * ?, seq_len, seq_len)
    bs = attention_mask.size(0)
    # (bs, (n_context + ?) * seq_len)
    attention_mask = torch.cat([attention_mask, self.encoder.attention_mask.max(1)[0].view(bs, -1)], dim=1)
    del self.encoder.attention_mask

  decoder_outputs = self.decoder(
    input_ids=decoder_input_ids,
    attention_mask=decoder_attention_mask,
    inputs_embeds=decoder_inputs_embeds,
    past_key_values=past_key_values,
    encoder_hidden_states=hidden_states,
    encoder_attention_mask=attention_mask,
    head_mask=decoder_head_mask,
    cross_attn_head_mask=cross_attn_head_mask,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
    gold_doc_dist=gold_doc_dist,
  )

  sequence_output = decoder_outputs[0]

  # Set device for model parallelism
  if self.model_parallel:
    torch.cuda.set_device(self.encoder.first_device)
    self.lm_head = self.lm_head.to(self.encoder.first_device)
    sequence_output = sequence_output.to(self.lm_head.weight.device)

  if self.config.tie_word_embeddings:
    # Rescale output before projecting on vocab
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    sequence_output = sequence_output * (self.model_dim ** -0.5)

  lm_logits = self.lm_head(sequence_output)

  loss = None
  if labels is not None:
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

  if not return_dict:
    output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
    return ((loss,) + output) if loss is not None else output

  return Seq2SeqLMOutput(
    loss=loss,
    logits=lm_logits,
    past_key_values=decoder_outputs.past_key_values,
    decoder_hidden_states=decoder_outputs.hidden_states,
    decoder_attentions=decoder_outputs.attentions,
    cross_attentions=decoder_outputs.cross_attentions,
    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
    encoder_hidden_states=encoder_outputs.hidden_states,
    encoder_attentions=encoder_outputs.attentions,
  )

T5ForConditionalGeneration.forward = t5forconditionalgeneration_forward

def t5layerselfattention_forward(
     self,
     hidden_states,
     attention_mask=None,
     position_bias=None,
     layer_head_mask=None,
     past_key_value=None,
     use_cache=False,
     output_attentions=False,
):
  if hasattr(self, 'additional_encode'):
    self.hidden_states = hidden_states
  normed_hidden_states = self.layer_norm(hidden_states)
  attention_output = self.SelfAttention(
    normed_hidden_states,
    mask=attention_mask,
    position_bias=position_bias,
    layer_head_mask=layer_head_mask,
    past_key_value=past_key_value,
    use_cache=use_cache,
    output_attentions=output_attentions,
  )
  if hasattr(self, 'additional_encode') and hidden_states.size(0) < attention_output[0].size(0):
    hidden_states = torch.cat([hidden_states, self.hidden_states], dim=0)
  hidden_states = hidden_states + self.dropout(attention_output[0])
  outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
  return outputs

T5LayerSelfAttention.forward = t5layerselfattention_forward

def t5stack_forward(
     self,
     input_ids=None,
     attention_mask=None,
     encoder_hidden_states=None,
     encoder_attention_mask=None,
     inputs_embeds=None,
     head_mask=None,
     cross_attn_head_mask=None,
     past_key_values=None,
     use_cache=None,
     output_attentions=None,
     output_hidden_states=None,
     return_dict=None,
     num_run_layers: int = None,  # number of layers to run (added parameters)
     gold_doc_dist = None,
):
  block_to_run = self.block if num_run_layers is None else self.block[:num_run_layers]

  # Model parallel
  if self.model_parallel:
    torch.cuda.set_device(self.first_device)
    self.embed_tokens = self.embed_tokens.to(self.first_device)
  use_cache = use_cache if use_cache is not None else self.config.use_cache
  output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
  output_hidden_states = (
    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
  )
  return_dict = return_dict if return_dict is not None else self.config.use_return_dict

  if input_ids is not None and inputs_embeds is not None:
    err_msg_prefix = "decoder_" if self.is_decoder else ""
    raise ValueError(
      f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
    )
  elif input_ids is not None:
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
  elif inputs_embeds is not None:
    input_shape = inputs_embeds.size()[:-1]
  else:
    err_msg_prefix = "decoder_" if self.is_decoder else ""
    raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

  if inputs_embeds is None:
    assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
    inputs_embeds = self.embed_tokens(input_ids)

  batch_size, seq_length = input_shape

  # required mask seq length can be calculated via length of past
  mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

  if use_cache is True:
    assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

  if attention_mask is None:
    attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
  if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
    encoder_seq_length = encoder_hidden_states.shape[1]
    encoder_attention_mask = torch.ones(
      batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
    )

  # initialize past_key_values with `None` if past does not exist
  if past_key_values is None:
    past_key_values = [None] * len(block_to_run)

  # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
  # ourselves in which case we just need to make it broadcastable to all heads.
  extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

  # If a 2D or 3D attention mask is provided for the cross-attention
  # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
  if self.is_decoder and encoder_hidden_states is not None:
    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    if encoder_attention_mask is None:
      encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
    encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
  else:
    encoder_extended_attention_mask = None

  # Prepare head mask if needed
  head_mask = self.get_head_mask(head_mask, self.config.num_layers)
  cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
  present_key_value_states = () if use_cache else None
  all_hidden_states = () if output_hidden_states else None
  all_attentions = () if output_attentions else None
  all_cross_attentions = () if (output_attentions and self.is_decoder) else None
  position_bias = None
  encoder_decoder_position_bias = None

  hidden_states = self.dropout(inputs_embeds)

  for i, (layer_module, past_key_value) in enumerate(zip(block_to_run, past_key_values)):
    layer_head_mask = head_mask[i]
    cross_attn_layer_head_mask = cross_attn_head_mask[i]
    # Model parallel
    if self.model_parallel:
      torch.cuda.set_device(hidden_states.device)
      # Ensure that attention_mask is always on the same device as hidden_states
      if attention_mask is not None:
        attention_mask = attention_mask.to(hidden_states.device)
      if position_bias is not None:
        position_bias = position_bias.to(hidden_states.device)
      if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
      if encoder_extended_attention_mask is not None:
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
      if encoder_decoder_position_bias is not None:
        encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
      if layer_head_mask is not None:
        layer_head_mask = layer_head_mask.to(hidden_states.device)
      if cross_attn_layer_head_mask is not None:
        cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if self.gradient_checkpointing and self.training:
      if use_cache:
        logger.warn(
          "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        )
        use_cache = False

      def create_custom_forward(module):
        def custom_forward(*inputs):
          return tuple(module(*inputs, use_cache, output_attentions))

        return custom_forward

      layer_outputs = checkpoint(
        create_custom_forward(layer_module),
        hidden_states,
        extended_attention_mask,
        position_bias,
        encoder_hidden_states,
        encoder_extended_attention_mask,
        encoder_decoder_position_bias,
        layer_head_mask,
        cross_attn_layer_head_mask,
        None,  # past_key_value is always None with gradient checkpointing
      )
    else:
      layer_outputs = layer_module(
        hidden_states,
        attention_mask=extended_attention_mask,
        position_bias=position_bias,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        encoder_decoder_position_bias=encoder_decoder_position_bias,
        layer_head_mask=layer_head_mask,
        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
      )
    if not self.is_decoder and hasattr(layer_module.module.layer[0].SelfAttention, 'preprocessed_mask'):
      extended_attention_mask = layer_module.module.layer[0].SelfAttention.preprocessed_mask
      del layer_module.module.layer[0].SelfAttention.preprocessed_mask

    # layer_outputs is a tuple with:
    # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
    if use_cache is False:
      layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

    hidden_states, present_key_value_state = layer_outputs[:2]

    # We share the position biases between the layers - the first layer store them
    # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
    # (cross-attention position bias), (cross-attention weights)
    position_bias = layer_outputs[2]
    if self.is_decoder and encoder_hidden_states is not None:
      encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
    # append next layer key value states
    if use_cache:
      present_key_value_states = present_key_value_states + (present_key_value_state,)

    if output_attentions:
      all_attentions = all_attentions + (layer_outputs[3],)
      if self.is_decoder:
        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

    # Model Parallel: If it's the last layer for that device, put things on the next device
    if self.model_parallel:
      for k, v in self.device_map.items():
        if i == v[-1] and "cuda:" + str(k) != self.last_device:
          hidden_states = hidden_states.to("cuda:" + str(k + 1))

  hidden_states = self.final_layer_norm(hidden_states)
  hidden_states = self.dropout(hidden_states)

  # Add last layer
  if output_hidden_states:
    all_hidden_states = all_hidden_states + (hidden_states,)

  if not return_dict:
    return tuple(
      v
      for v in [
        hidden_states,
        present_key_value_states,
        all_hidden_states,
        all_attentions,
        all_cross_attentions,
      ]
      if v is not None
    )
  return BaseModelOutputWithPastAndCrossAttentions(
    last_hidden_state=hidden_states,
    past_key_values=present_key_value_states,
    hidden_states=all_hidden_states,
    attentions=all_attentions,
    cross_attentions=all_cross_attentions,
  )

T5Stack.forward = t5stack_forward

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
    bs = states.size(0)
    return states.view(bs, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

  def unshape(states):
    """reshape"""
    bs = states.size(0)
    return states.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

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

  # separate position_bias from mask
  # TODO: any better workaround?
  scores += position_bias

  # collect encoder attn (before applying mask)
  if hasattr(self, 'collect_for_retrieval'):
    collect_return = self.collect_for_retrieval(scores, hidden_states, query_states, key_states, value_states, position_bias, mask)
    if collect_return is not None:  # update these tensors to make them larger
      scores, query_states, key_states, value_states, mask = collect_return

  if mask is not None:
    # apply token-leve mask
    scores = scores + mask  # (batch_size, n_heads, seq_length, key_length)
    # compute the KL between encoder and decoder attn
    if self.is_decoder and hasattr(self, 'encoder_decoder_kl'):
      self.kl_loss = self.encoder_decoder_kl(scores, mask.eq(0))
    # modify decoder attn based on encoder
    elif self.is_decoder and hasattr(self, 'combine_encoder_decoder_attn'):
      if hasattr(self, 'decoder_attn_ctx_normalize'):  # normalize raw attn wrt ctx
        scores = self.decoder_attn_ctx_normalize(scores)
      scores = self.combine_encoder_decoder_attn(scores)

  attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    scores
  )  # (batch_size, n_heads, seq_length, key_length)
  attn_weights = nn.functional.dropout(
    attn_weights, p=self.dropout, training=self.training
  )  # (batch_size, n_heads, seq_length, key_length)

  # collect decoder attn (after applying mask)
  if self.is_decoder and not self.training and hasattr(self, 'collect_cross_attention'):
    self.collect_cross_attention(scores, mask.eq(0))

  # Mask heads if we want to
  if layer_head_mask is not None:
    attn_weights = attn_weights * layer_head_mask

  attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
  attn_output = self.o(attn_output)

  present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
  outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

  need_to_push_attn_in_output_to_enable_bp = \
    self.training and hasattr(self, 'collect_for_retrieval') and not hasattr(self, 'just_collect')

  if output_attentions or need_to_push_attn_in_output_to_enable_bp:
    if need_to_push_attn_in_output_to_enable_bp:
      outputs = outputs + (self.retrieval['two_tower_attn_score_full']
                           if 'two_tower_attn_score_full' in self.retrieval
                           else self.retrieval['two_tower_attn_score'],)  # (bs, num_heads)
    else:
      outputs = outputs + (attn_weights,)
  return outputs

T5Attention.forward = t5attention_forward

def fid_run(model, input_ids=None, attention_mask=None, accumulate_steps: int = None, **kwargs):
  if not accumulate_steps:
    result = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    loss = result[0]
    loss.backward()
    return result
  
  # split tensors into accumulation_steps parts
  bsz = input_ids.size(0)
  assert bsz == attention_mask.size(0)
  assert bsz % accumulate_steps == 0, 'batch size not dividable by accumulate_steps'
  _input_ids = input_ids.chunk(accumulate_steps)
  _attention_mask = attention_mask.chunk(accumulate_steps)
  _kwargs = {}
  _rand_states = []
  for k in ['attention_separate_mask', 'input_doc_ids', 'gold_doc_dist', 'labels']:
    if k in kwargs:
      if type(kwargs[k]) is torch.Tensor:
        assert kwargs[k].size(0) == bsz
        _kwargs[k] = kwargs[k].chunk(accumulate_steps)
      elif type(kwargs[k]) is np.ndarray:
        assert kwargs[k].shape[0] == bsz
        _kwargs[k] = np.split(kwargs[k], accumulate_steps)
      else:
        raise ValueError
  def get_kwargs_chunk(step):
    kwargs_chunk = {k: v for k, v in kwargs.items()}
    for k in _kwargs:
      kwargs_chunk[k] = _kwargs[k][step]
    return kwargs_chunk

  # TODO: not applicable to multiple layers
  # run bi-encoder to get representations in training mode w/o gradient
  accumulated = {
    'total': accumulate_steps,
    'query_states': [], 
    'key_states': [],
    'attention_mask': [],
    'decoder_attention': []
  }
  with torch.no_grad():
    for acc_step in range(accumulate_steps):
      _rand_states.append(RandContext(*RandContext.get_input_tensors(
        input_ids=_input_ids[acc_step], 
        attention_mask=_attention_mask[acc_step], 
        **get_kwargs_chunk(acc_step))))
      enc_reg_point, use_head_idx, dec_reg_point = model(
        input_ids=_input_ids[acc_step], 
        attention_mask=_attention_mask[acc_step], 
        **get_kwargs_chunk(acc_step), 
        collect_necessary_for_ibn=True)
      assert use_head_idx is not None
      qs, ks, am = enc_reg_point.retrieval['query_states'], enc_reg_point.retrieval['key_states'], enc_reg_point.retrieval['attention_mask']
      da = dec_reg_point.retrieval['decoder_attention']
      global_q, global_k, global_attn, global_dec_attn = all_gather_tensors(
        qs[:, use_head_idx:use_head_idx + 1].contiguous(), ks[:, use_head_idx:use_head_idx + 1].contiguous(), am, da)
      accumulated['query_states'].extend(global_q)
      accumulated['key_states'].extend(global_k)
      accumulated['attention_mask'].extend(global_attn)
      accumulated['decoder_attention'].extend(global_dec_attn)
  
  # run full model w/ gradient
  # inform encoder and decoder
  for acc_step in range(accumulate_steps):
    state = _rand_states[acc_step]
    accumulated['step'] = acc_step
    with state:
      result = model(
        input_ids=_input_ids[acc_step], 
        attention_mask=_attention_mask[acc_step], 
        **get_kwargs_chunk(acc_step),
        accumulated=accumulated)
      loss = result[0]
      loss.backward()
  return result

class FiDT5Config(transformers.T5Config):
  def __init__(self,
               *args,
               n_layer_two_tower: int = 0,
               layer_for_retrieval: str = 'first',
               attention_mask: str = 'separate',
               retrieval_aggregation_method: str = 'all-avg-max',
               query_in_decoder: str = 'no',
               num_keep_ctx_in_decoder: int = 0,
               combine_weight: float = 0.0,
               only_topk_n_context: int = 0,
               keep_ctx_in_decoder_with_head: int = None,
               keep_ctx_in_decoder_head_tau: float = 1.0,
               head_weights_norm_func: str = 'softmax',
               encoder_attention_pre_softmax: bool = False,
               encoder_decoder_kl_ratio: float = 0,
               encoder_decoder_kl_method: str = 'merge',
               encoder_encoder_kl_method: str = None,
               in_batch_negative: bool = False,
               in_batch_negative_size: int = 0,
               in_batch_negative_max_num_query: int = None,
               pairwise_loss: str = None,
               memory_bank: int = 0,
               memory_bank_topk: int = 0,
               memory_use_random: bool = False,
               memory_bank_recompute: bool = False,
               memory_bank_additional_encode: bool = False,
               encoder_encoder_kl_ratio: float = 0,
               encoder_encoder_kl: str = None,
               encoder_encoder_kl_sparsity: int = 0,
               decoder_attn_ctx_normalize: bool = False,
               max_over_head: bool = False,
               term_weight_parameter: bool = False,
               embedding_normalize: bool = False,
               use_gold_doc_dist: bool = False,
               retrieval_projection: str = None,
               kl_loss_reduction: str = None,
               no_qa: bool = False,
               n_context_for_ibn: int = None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.n_layer_two_tower = n_layer_two_tower
    self.layer_for_retrieval = layer_for_retrieval
    self.attention_mask = attention_mask
    self.retrieval_aggregation_method = retrieval_aggregation_method
    self.query_in_decoder = query_in_decoder
    self.num_keep_ctx_in_decoder = num_keep_ctx_in_decoder
    self.combine_weight = combine_weight
    self.only_topk_n_context = only_topk_n_context
    self.keep_ctx_in_decoder_with_head = keep_ctx_in_decoder_with_head
    self.keep_ctx_in_decoder_head_tau = keep_ctx_in_decoder_head_tau
    self.head_weights_norm_func = head_weights_norm_func
    self.encoder_attention_pre_softmax = encoder_attention_pre_softmax
    self.encoder_decoder_kl_ratio = encoder_decoder_kl_ratio
    self.encoder_decoder_kl_method = encoder_decoder_kl_method
    self.encoder_encoder_kl_method = encoder_encoder_kl_method
    self.in_batch_negative = in_batch_negative
    self.in_batch_negative_size = in_batch_negative_size
    self.in_batch_negative_max_num_query = in_batch_negative_max_num_query
    self.pairwise_loss = pairwise_loss
    self.memory_bank = memory_bank
    self.memory_bank_topk = memory_bank_topk
    self.memory_use_random = memory_use_random
    self.memory_bank_recompute = memory_bank_recompute
    self.memory_bank_additional_encode = memory_bank_additional_encode
    self.encoder_encoder_kl_ratio = encoder_encoder_kl_ratio
    self.encoder_encoder_kl = encoder_encoder_kl
    self.encoder_encoder_kl_sparsity = encoder_encoder_kl_sparsity
    self.decoder_attn_ctx_normalize = decoder_attn_ctx_normalize
    self.max_over_head = max_over_head
    self.term_weight_parameter = term_weight_parameter
    self.embedding_normalize = embedding_normalize
    self.use_gold_doc_dist = use_gold_doc_dist
    self.retrieval_projection = retrieval_projection
    self.kl_loss_reduction = kl_loss_reduction
    self.no_qa = no_qa
    self.n_context_for_ibn = n_context_for_ibn

class FiDT5(transformers.T5ForConditionalGeneration):
    config_class = FiDT5Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wrap_encoder(config)
        self.wrap_decoder(config)
        self.collect_kl_loss_from_decoder = bool(config.encoder_decoder_kl_ratio)
        self.collect_kl_loss_from_encoder = bool(config.encoder_encoder_kl_ratio)
        self.n_layer_two_tower = config.n_layer_two_tower
        self.num_heads = config.num_heads
        self.max_over_head = config.max_over_head
        self.memory_bank_recompute = config.memory_bank_recompute
        self.no_qa = config.no_qa

    @classmethod
    def from_t5(cls,
                model_name: str,
                n_layer_two_tower: int = 0,
                layer_for_retrieval: str = 'first',
                attention_mask: str = 'separate',
                retrieval_aggregation_method: str = 'all-avg-max',
                query_in_decoder: str = 'no',
                num_keep_ctx_in_decoder: int = 0,
                combine_weight: float = 0.0,
                only_topk_n_context: int = 0,
                keep_ctx_in_decoder_with_head: int = None,
                keep_ctx_in_decoder_head_tau: float = 1.0,
                head_weights_norm_func: str = 'softmax',
                encoder_attention_pre_softmax: bool = False,
                encoder_decoder_kl_ratio: float = 0,
                encoder_decoder_kl_method: str = 'merge',
                encoder_encoder_kl_method: str = None,
                in_batch_negative: bool = False,
                in_batch_negative_size: int = 0,
                in_batch_negative_max_num_query: int = None,
                pairwise_loss: str = None,
                memory_bank: int = 0,
                memory_bank_topk: int = 0,
                memory_use_random: bool = False,
                memory_bank_recompute: bool = False,
                memory_bank_additional_encode: bool = False,
                encoder_encoder_kl_ratio: float = 0,
                encoder_encoder_kl: str = None,
                encoder_encoder_kl_sparsity: int = 0,
                decoder_attn_ctx_normalize: bool = False,
                max_over_head: bool = False,
                term_weight_parameter: bool = False,
                embedding_normalize: bool = False,
                use_gold_doc_dist: bool = False,
                retrieval_projection: str = None,
                kl_loss_reduction: str = None,
                no_qa: bool = False,
                n_context_for_ibn: int = None):
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        config = cls.config_class(
          n_layer_two_tower=n_layer_two_tower,
          layer_for_retrieval=layer_for_retrieval,
          attention_mask=attention_mask,
          retrieval_aggregation_method=retrieval_aggregation_method,
          query_in_decoder=query_in_decoder,
          num_keep_ctx_in_decoder=num_keep_ctx_in_decoder,
          combine_weight=combine_weight,
          only_topk_n_context=only_topk_n_context,
          keep_ctx_in_decoder_with_head=keep_ctx_in_decoder_with_head,
          keep_ctx_in_decoder_head_tau=keep_ctx_in_decoder_head_tau,
          head_weights_norm_func=head_weights_norm_func,
          encoder_attention_pre_softmax=encoder_attention_pre_softmax,
          encoder_decoder_kl_ratio=encoder_decoder_kl_ratio,
          encoder_decoder_kl_method=encoder_decoder_kl_method,
          encoder_encoder_kl_method=encoder_encoder_kl_method,
          in_batch_negative=in_batch_negative,
          in_batch_negative_size=in_batch_negative_size,
          in_batch_negative_max_num_query=in_batch_negative_max_num_query,
          pairwise_loss=pairwise_loss,
          memory_bank=memory_bank,
          memory_bank_topk=memory_bank_topk,
          memory_use_random=memory_use_random,
          memory_bank_recompute=memory_bank_recompute,
          memory_bank_additional_encode=memory_bank_additional_encode,
          encoder_encoder_kl_ratio=encoder_encoder_kl_ratio,
          encoder_encoder_kl=encoder_encoder_kl,
          encoder_encoder_kl_sparsity=encoder_encoder_kl_sparsity,
          decoder_attn_ctx_normalize=decoder_attn_ctx_normalize,
          max_over_head=max_over_head,
          term_weight_parameter=term_weight_parameter,
          embedding_normalize=embedding_normalize,
          use_gold_doc_dist=use_gold_doc_dist,
          retrieval_projection=retrieval_projection,
          kl_loss_reduction=kl_loss_reduction,
          no_qa=no_qa,
          n_context_for_ibn=n_context_for_ibn,
          **t5.config.to_dict())
        model = cls(config)
        model.load_t5(t5.state_dict())
        return model

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, accumulated=None, **kwargs):
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
        
        collect_necessary_for_ibn = False
        if 'collect_necessary_for_ibn' in kwargs:
          collect_necessary_for_ibn = kwargs['collect_necessary_for_ibn']
          del kwargs['collect_necessary_for_ibn']

        if accumulated is not None:  # pass it to encoder and decoder
            self.encoder.set_accumulated(accumulated)
            self.decoder.set_accumulated(accumulated)
        elif hasattr(self.encoder, 'delete_accumulated') and hasattr(self.decoder, 'delete_accumulated'):
            self.encoder.delete_accumulated()
            self.decoder.delete_accumulated()
        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        if accumulated is not None:
            self.encoder.delete_accumulated()
            self.decoder.delete_accumulated()

        if 'only_bi_encoder_forward' in kwargs and kwargs['only_bi_encoder_forward']:
          return result
        if collect_necessary_for_ibn:
          enc_reg = self.encoder.get_reg_point()
          dec_reg = self.decoder.get_reg_point()
          return enc_reg, self.encoder.use_head_idx, dec_reg

        kl_loss = 0
        if self.collect_kl_loss_from_decoder:
          kl_loss += self.decoder.get_kl_loss()
        if self.collect_kl_loss_from_encoder:
          kl_loss += self.encoder.get_kl_loss()
        qa_loss_factor = 0.0 if self.no_qa else 1.0
        if accumulated is not None:
          qa_loss_factor /= accumulated['total']  # average across accumulations
        if 'return_dict' in kwargs and kwargs['return_dict'] and result.loss is not None:
          result.loss = qa_loss_factor * result.loss + kl_loss
        elif 'labels' in kwargs and kwargs['labels'] is not None:
          result = (qa_loss_factor * result[0] + kl_loss,) + result[1:]
        self.reset_attention_separate_mask()  # always reset separate mask to clean it up
        return result

    def encode_context(self,
                       input_ids,  # (num_docs, seq_len)
                       attention_mask,  # (num_docs, seq_len)
                       max_query_len: int = None):
        self.encoder(input_ids=input_ids, attention_mask=attention_mask, direct=True)
        # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head)
        context_embedding = self.encoder.get_collected_for_retrieval()['key_states']
        # (n_heads, max_seq_len, max_seq_len)
        position_bias = self.encoder.encoder.block[0].module.layer[0].SelfAttention.compute_bias(1500, 1500)[0]  # larger than 1024 to avoid overflow
        if self.config.keep_ctx_in_decoder_with_head is not None:
          hi = self.config.keep_ctx_in_decoder_with_head
          context_embedding = context_embedding[:, :, hi:hi + 1]  # (num_docs, n_layer, 1 (n_heads), seq_len, emb_size_per_head)
          position_bias = position_bias[hi:hi + 1]  # (1 (n_heads), max_seq_len, max_seq_len)
        if max_query_len:  # use position bias as additional "embedding"
          num_docs, n_layer, n_heads, seq_len = context_embedding.size()[:4]
          # (n_heads, seq_len, max_query_len)
          position_bias = position_bias[:, :max_query_len, max_query_len:max_query_len + seq_len].permute([0, 2, 1])
          assert position_bias.size(1) == seq_len
          # (num_docs, n_layer, n_heads, seq_len, emb_size_per_head + max_query_len)
          context_embedding = torch.cat(
            [context_embedding, position_bias.unsqueeze(0).unsqueeze(0).repeat(num_docs, n_layer, 1, 1, 1)], dim=-1)
        return context_embedding

    def encode_query(self,
                     input_ids,  # (num_queries, seq_len)
                     attention_mask,  # (num_queries, seq_len)
                     max_query_len: int = None,
                     random_position: bool = False):
        self.encoder(input_ids=input_ids, attention_mask=attention_mask, direct=True)
        collect = self.encoder.get_collected_for_retrieval()
        query_embedding = collect['query_states']  # (num_queries, n_layer, n_heads, seq_len, emb_size_per_head)
        term_weights = collect['term_weights'] if 'term_weights' in collect else None  # (num_queries, n_layer, n_heads, seq_len)
        if self.config.keep_ctx_in_decoder_with_head is not None:
          hi = self.config.keep_ctx_in_decoder_with_head
          query_embedding = query_embedding[:, :, hi:hi + 1]  # (num_queries, n_layer, 1 (n_heads), seq_len, emb_size_per_head)
          if term_weights is not None:
            term_weights = term_weights[:, :, hi:hi + 1]  # (num_queries, n_layer, 1 (n_heads), seq_len)
        if max_query_len:
          num_queries, n_layer, n_heads, seq_len = query_embedding.size()[:4]
          assert seq_len <= max_query_len
          # get index
          if random_position:
            offset_to_last_adj = torch.randint(0, max_query_len, (num_queries, seq_len)).to(input_ids.device)
          else:
            pos_ind = torch.cumsum(attention_mask, dim=-1) - 1  # [0, 1, 2, 3, 3, ..., 3] a 4-token query
            query_len = torch.sum(attention_mask, dim=-1)  # 4
            offset_to_last = pos_ind - query_len.unsqueeze(-1)  # [-4, -3, -2, -1, -1, ..., -1]
            offset_to_last_adj = max_query_len + offset_to_last
          # convert to one-hot and concat
          one_hot = torch.zeros_like(input_ids).unsqueeze(-1).repeat(1, 1, max_query_len)  # (num_queries, seq_len, max_query_len)
          one_hot.scatter_(-1, offset_to_last_adj.unsqueeze(-1), 1)
          # (num_queries, n_layer, n_heads, seq_len, emb_size_per_head + max_query_len)
          query_embedding = torch.cat(
            [query_embedding, one_hot.unsqueeze(1).unsqueeze(1).repeat(1, n_layer, n_heads, 1, 1)], dim=-1)
        return query_embedding, term_weights

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
      def gcif():
        r = self.get_collected_for_retrieval()
        ttas = r['two_tower_attn_score_full'] if 'two_tower_attn_score_full' in r else r['two_tower_attn_score']
        ttasfm = r['two_tower_attn_score_full_mask'] if 'two_tower_attn_score_full' in r else None
        n_context = r['n_context']
        return ttas, ttasfm, n_context
      self.decoder = DecoderWrapper(config, self.decoder, gcif)

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
      self.encoder.load_state_dict(model.encoder.state_dict(), strict=False)
      if type(self.decoder) == type(model.decoder) == DecoderWrapper:
        self.decoder.load_state_dict(model.decoder.state_dict())
      else:
        self.get_inner_decoder().load_state_dict(model.get_inner_decoder().state_dict())

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        all_ckpt = lambda i: True  # checkpoint all layers
        later_ckpt = lambda i: i > self.n_layer_two_tower  # only checkpoint layers after bi-encoder
        use_ckpt_per_layer = all_ckpt
        for i, layer in enumerate(self.encoder.encoder.block):
            layer.use_checkpoint = use_checkpoint and use_ckpt_per_layer(i)

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

      encoder_scores = None
      if type(self.decoder) is DecoderWrapper:
        encoder_scores = self.decoder.encoder_imp_agg  # batch_size, n_passages

      return scores, encoder_scores

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
                attn.collect_cross_attention = types.MethodType(functools.partial(
                  collect_cross_attention, n_context=n_context, sum_over_tokens=True, use_softmax=True), attn)

class DecoderWrapper(torch.nn.Module):
  def __init__(self, config, decoder, get_encoder_importance_func: Callable):
    super().__init__()
    self.decoder = decoder
    self.get_encoder_importance_func = get_encoder_importance_func
    self.num_keep_ctx_in_decoder = config.num_keep_ctx_in_decoder
    self.keep_ctx_in_decoder_with_head = config.keep_ctx_in_decoder_with_head
    self.keep_ctx_in_decoder_head_tau = config.keep_ctx_in_decoder_head_tau
    self.encoder_decoder_kl_ratio = config.encoder_decoder_kl_ratio
    self.decoder_attn_ctx_normalize = config.decoder_attn_ctx_normalize
    self.only_topk_n_context = config.only_topk_n_context
    self.layer_for_retrieval = config.layer_for_retrieval
    self.encoder_decoder_kl_method = config.encoder_decoder_kl_method
    self.encoder_encoder_kl_method = config.encoder_encoder_kl_method
    self.n_layer_two_tower = config.n_layer_two_tower
    self.encoder_attention_pre_softmax = config.encoder_attention_pre_softmax
    self.in_batch_negative = config.in_batch_negative
    self.pairwise_loss = config.pairwise_loss
    self.memory_bank_topk = config.memory_bank_topk
    self.memory_bank_additional_encode = config.memory_bank_additional_encode
    self.max_over_head = config.max_over_head
    self.use_gold_doc_dist = config.use_gold_doc_dist
    self.retrieval_projection = config.retrieval_projection
    self.kl_loss_reduction = config.kl_loss_reduction
    if self.encoder_attention_pre_softmax and self.num_keep_ctx_in_decoder:
      raise NotImplementedError
    if self.encoder_decoder_kl_ratio and self.num_keep_ctx_in_decoder:
      raise ValueError('only one of the KL and combined loss should be used')
    if self.decoder_attn_ctx_normalize:
      assert self.num_keep_ctx_in_decoder and not self.encoder_decoder_kl_ratio, \
        'normalized decoder is not used in a proper setting'
    
    if FiDT5.need_wrap_decoder(config) and not self.max_over_head and \
      (not self.retrieval_projection or not self.retrieval_projection.startswith('hidden')):
      if config.keep_ctx_in_decoder_with_head is None:
        if self.encoder_decoder_kl_method == 'merge':
          if self.layer_for_retrieval in {'first', 'emb'}:
            nw = config.num_heads
          elif self.layer_for_retrieval == 'emb-first':
            nw = 2 * config.num_heads
          elif self.layer_for_retrieval == 'prev-first':
            nw = (self.n_layer_two_tower + 1) * config.num_heads
          elif self.layer_for_retrieval == 'after-first':
            nw = (config.num_layers - self.n_layer_two_tower) * config.num_heads
          elif self.layer_for_retrieval == 'last-first':
            nw = 2 * config.num_heads
          else:
            raise NotImplementedError
          self.head_weights = torch.nn.Parameter(torch.zeros(nw), requires_grad=True)  # merge considered layers
        elif self.encoder_decoder_kl_method in {'separate', 'cross'}:
          if self.layer_for_retrieval in {'first', 'emb'}:
            nl = 1
          elif self.layer_for_retrieval in {'emb-first', 'last-first'}:
            nl = 2
          elif self.layer_for_retrieval == 'prev-first':
            nl = self.n_layer_two_tower + 1
          elif self.layer_for_retrieval == 'after-first':
            nl = config.num_layers - self.n_layer_two_tower
          else:
            raise NotImplementedError
          self.head_weights = torch.nn.Parameter(torch.zeros(nl, config.num_heads), requires_grad=True)  # use different layers separately
        else:
          raise NotImplementedError
      else:
        assert self.layer_for_retrieval == 'first', \
          'only the first layer after bi-encoder should used for retrieval when using a specific head'
        weights = [-1e5] * config.num_heads
        weights[config.keep_ctx_in_decoder_with_head] = 1.0
        self.head_weights = torch.nn.Parameter(torch.tensor(weights), requires_grad=False)
    
    if FiDT5.need_wrap_decoder(config):
      if config.head_weights_norm_func == 'softmax':
        self.head_weights_norm_func = lambda x: torch.softmax(x / self.keep_ctx_in_decoder_head_tau, -1)
      elif config.head_weights_norm_func == 'sparsemax':
        self.head_weights_norm_func = lambda x: sparsemax(x, -1)
      else:
        raise NotImplementedError
      self.init_combine_weight = config.combine_weight
      self.combine_weight = torch.nn.Parameter(torch.tensor(self.init_combine_weight), requires_grad=True)

  def get_kl_loss(self):
    assert self.encoder_decoder_kl_ratio
    world_size = 1
    if self.in_batch_negative and global_context['opt'].is_distributed:  # to counter the gradient avg across gpus
      world_size = global_context['opt'].world_size
    kl_loss = self.get_reg_point().kl_loss * self.encoder_decoder_kl_ratio * world_size
    return kl_loss
  
  def get_reg_point(self):
    return self.decoder.block[-1].layer[1].EncDecAttention
  
  def set_accumulated(self, accumulated: Dict):
    self.accumulated = accumulated
  
  def delete_accumulated(self):
    if hasattr(self, 'accumulated'):
      del self.accumulated

  def forward(
       self,
       input_ids=None,
       attention_mask=None,
       encoder_hidden_states=None,
       encoder_attention_mask=None,
       gold_doc_dist=None,
       **kwargs):
    # fetch encoder importance
    # (num_q, num_d, num_layer, num_head, [num_toks]), (num_q, num_d, num_toks)
    encoder_imp, encoder_imp_tok_mask, n_context = self.get_encoder_importance_func()
    has_token = encoder_imp.dim() == 5
    num_q, num_d, num_layer, num_head = encoder_imp.size()[:4]
    num_toks = encoder_imp.size(4) if has_token else None
    dt = encoder_hidden_states.size(1)  # n_context * ctx_len
    assert dt % n_context == 0, 'encoder_hidden_states shape error'
    ctx_len = dt // n_context

    # apply combine weight
    encoder_imp = torch.exp(self.combine_weight) * encoder_imp
    WandbLogger.log_w_step({'combine-weight': torch.exp(self.combine_weight).item()})

    # softmax
    if self.encoder_attention_pre_softmax:
      assert not has_token, 'pre softmax might not be good for token-level loss'
      encoder_imp = torch.log_softmax(encoder_imp, dim=1)

    # reshape
    if self.encoder_decoder_kl_method == 'merge':
      if self.layer_for_retrieval in {'first', 'emb'}:  # use the first layer
        encoder_imp = encoder_imp[:, :, 0]  # (num_q, num_d, num_head, [num_toks])
      elif self.layer_for_retrieval in {'emb-first', 'last-first'}:  # use the emb/last layer and first layer after bi-encoder
        assert encoder_imp.size(2) == 2, 'provide attn for emb and the first layer after bi-encoder'
        v = (num_q, num_d, -1) if not has_token else (num_q, num_d, -1, num_toks)
        encoder_imp = encoder_imp.view(*v)  # (num_q, num_d, 2 * num_head, [num_toks])
      elif self.layer_for_retrieval in {'prev-first', 'after-first'}:  # use all layers before first and after first
        v = (num_q, num_d, -1) if not has_token else (num_q, num_d, -1, num_toks)
        encoder_imp = encoder_imp.view(*v)  # (num_q, num_d, ? * num_head, [num_toks])
      else:
        raise NotImplementedError

      if num_head == 1:
        if has_token:
          raise NotImplementedError
        encoder_imp = encoder_imp[:, :, 0]  # (num_q, num_d)
      else:
        # combine multiple heads
        hwn = self.head_weights_norm_func(self.head_weights)
        WandbLogger.log_w_step({'head-weight': hwn})
        if self.encoder_attention_pre_softmax:
          assert not has_token, 'pre softmax might not be good for token-level loss'
          encoder_imp = torch.logsumexp(encoder_imp + torch.log(torch.clamp(hwn, min=1e-10))[None, None, :], dim=-1)
        else:
          if not has_token:
            encoder_imp = (encoder_imp * hwn[None, None, :]).sum(-1)  # (num_q, num_d)
          else:
            encoder_imp = (encoder_imp * hwn[None, None, :, None]).sum(-2)  # (num_q, num_d, num_toks)
            # softmax over all toks in all context
            encoder_imp = encoder_imp * encoder_imp_tok_mask - ~encoder_imp_tok_mask * 1e5
            encoder_imp = torch.log_softmax(encoder_imp.view(num_q, -1), dim=-1).view(num_q, num_d, num_toks)

    elif self.encoder_decoder_kl_method in {'separate', 'cross'}:
      # combine each layer separately
      hwn = self.head_weights_norm_func(self.head_weights)
      for li in range(hwn.size(0)):
        WandbLogger.log_w_step({'head-weight' + ('' if li == 0 else f'-L{li}'): hwn[li]})
      if self.encoder_attention_pre_softmax:
        raise NotImplementedError
      else:
        if not has_token:
          encoder_imp = (encoder_imp * hwn[None, None, :, :]).sum(-1)  # (num_q, num_d, num_layer)
          encoder_imp = encoder_imp.permute(2, 0, 1)  # (num_layer, num_q, num_d)
        else:
          raise NotImplementedError
    else:
      raise NotImplementedError

    # used for output
    if not has_token:
      self.encoder_imp_agg = encoder_imp
    else:
      self.encoder_imp_agg = (torch.exp(encoder_imp) * encoder_imp_tok_mask).sum(-1)  # (num_q, num_d)

    if self.num_keep_ctx_in_decoder:
      if has_token:
        raise NotImplementedError
      # spasify
      assert self.num_keep_ctx_in_decoder <= n_context
      value, ind = encoder_imp.topk(self.num_keep_ctx_in_decoder, -1)
      encoder_imp_mask = torch.zeros_like(encoder_imp).bool()
      encoder_imp_mask.scatter_(-1, ind, True)  # encoder_imp still contains all values and we use this mask to force sparsity
      # reshape to (num_q, 1 (num_head), 1 (query_len), num_d * ctx_len)
      encoder_imp = encoder_imp.unsqueeze(-1).repeat(1, 1, ctx_len).view(num_q, 1, 1, -1)
      # convert to final mask of shape (num_q, 1 (num_head), 1 (query_len), num_d * ctx_len)
      encoder_imp_mask = self.decoder.invert_attention_mask(encoder_imp_mask.unsqueeze(-1).repeat(1, 1, ctx_len).view(num_q, -1))
      # assign to each cross attn module
      for layer in self.decoder.block:
        reg_point = layer.layer[1].EncDecAttention
        reg_point.combine_encoder_decoder_attn = lambda dec_score: dec_score + encoder_imp + encoder_imp_mask
        if self.decoder_attn_ctx_normalize:
          reg_point.decoder_attn_ctx_normalize = types.MethodType(
            functools.partial(decoder_attn_ctx_normalize, n_context=num_d), reg_point)
    elif self.encoder_decoder_kl_ratio:
      # assign to the last cross attn module
      reg_point = self.get_reg_point()
      reg_point.encoder_decoder_kl = types.MethodType(functools.partial(
        encoder_decoder_kl,
        encoder_score=encoder_imp,
        encoder_score_mask=encoder_imp_tok_mask,
        gold_doc_dist=gold_doc_dist,
        n_context=n_context,
        only_topk_n_context=self.only_topk_n_context,
        use_softmax=True,
        encoder_score_pre_softmaxed=self.encoder_attention_pre_softmax,
        in_batch_negative=self.in_batch_negative and self.training,
        pairwise_loss=self.pairwise_loss,
        memory_bank_topk=self.memory_bank_topk if self.training else 0,
        memory_bank_additional_encode=self.memory_bank_additional_encode if self.training else 0,
        use_gold_doc_dist=self.use_gold_doc_dist,
        kl_loss_reduction=self.kl_loss_reduction,
        encoder_decoder_kl_method=self.encoder_decoder_kl_method,
        encoder_encoder_kl_method=self.encoder_encoder_kl_method,
        accumulated=self.accumulated if hasattr(self, 'accumulated') else None,
      ), reg_point)
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
        self.num_heads = config.num_heads
        self.n_layer_two_tower = config.n_layer_two_tower
        self.layer_for_retrieval = config.layer_for_retrieval
        self.encoder_encoder_kl_ratio = config.encoder_encoder_kl_ratio
        self.encoder_encoder_kl = config.encoder_encoder_kl
        self.encoder_encoder_kl_sparsity = config.encoder_encoder_kl_sparsity
        self.memory_bank_topk = config.memory_bank_topk
        self.memory_use_random = config.memory_use_random
        self.memory_bank_additional_encode = config.memory_bank_additional_encode
        self.max_over_head = config.max_over_head
        self.use_head_idx = get_single_head_idx(self.num_heads, self.n_layer_two_tower, self.max_over_head)
        self.pad_token_id = config.pad_token_id
        self.memory_bank_inference = True  # use inference mode to run memory bank
        self.separate_process = True  # use another process to handle memory bank
        # setup memory bank directly or a process to handle memory bank
        self.memory_bank = None
        if config.memory_bank:
          if not hasattr(global_context['opt'], 'memory_bank_gpu'):
            use_gpu = False
          elif global_context['opt'].memory_bank_gpu is None:
            use_gpu = True if not self.separate_process else torch.cuda.device_count() - 1  # use the last gpu when using a separate process for memory bank
          else:
            use_gpu = list(map(int, global_context['opt'].memory_bank_gpu.split(',')))
          init_kwargs={
            'max_size': config.memory_bank, 'indexing_dimension': config.d_kv, 'use_gpu': use_gpu, 
            'use_head_idx': self.use_head_idx, 'pad_token_id': self.pad_token_id, 
            'bank_topk': self.memory_bank_topk, 'use_random': self.memory_use_random}
          self.memory_bank = MemoryBankProcessHelper(**init_kwargs) if self.separate_process else MemoryBank(**init_kwargs)
          if self.memory_bank_inference:
            assert config.memory_bank_recompute and self.memory_bank_additional_encode, 'not implemented'
        def bi_encoder_forward(
             input_ids,  # (bs, seq_len)
             attention_mask):  # (bs, seq_len, seq_len)
          reg_point = self.get_reg_point()
          reg_point.just_collect = True
          _ = self.encoder(input_ids, attention_mask, num_run_layers=self.n_layer_two_tower + 1)
          del reg_point.just_collect
          return reg_point, self.use_head_idx
        self.bi_encoder_forward = bi_encoder_forward
        self.apply_t5block_wrapper(config)
    
    def get_reg_point(self):
      return self.encoder.block[self.n_layer_two_tower].module.layer[0].SelfAttention

    def forward(self, input_ids=None, attention_mask=None, input_doc_ids=None, **kwargs):
        if 'direct' in kwargs and kwargs['direct']:  # no reshaping, used in retrieval
          del kwargs['direct']
          return self.bi_encoder_forward(input_ids, attention_mask)

        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        if input_doc_ids is not None:
            input_doc_ids = input_doc_ids.reshape(-1)  # (bs * n_passage)
        if hasattr(self, 'attention_separate_mask') and self.attention_separate_mask is not None:  # use separate attention mask
          attention_mask = self.attention_separate_mask.view(
            *((bsz * self.n_passages,) + self.attention_separate_mask.size()[2:]))  # (bs * n_passage, seq_len * seq_len)
        else:
          attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)  # (bs * n_passage, seq_len)

        use_memory_bank = self.training and self.memory_bank_inference and self.memory_bank is not None
        use_memory_bank_and_retrieve = use_memory_bank and self.memory_bank_topk and len(self.memory_bank) >= (self.memory_bank_topk + self.n_passages)
        with torch.no_grad():
          was_training = self.training
          self.eval()
          if use_memory_bank:  # generate bi-encoder representations
            n_context = self.n_passages
            collected = self.bi_encoder_forward(input_ids, attention_mask)[0].retrieval
            query_states, key_states = collected['query_states'], collected['key_states']
            if self.separate_process:
              self.memory_bank.start()
          if use_memory_bank_and_retrieve:  # retrieve, store, then update the input_ids and attention_mask
            _input_ids, _attention_mask = self.memory_bank.query_then_add_from_t5(
              query_states, key_states, attention_mask, input_ids, input_doc_ids, n_context=n_context)[:2]
            if self.memory_bank_additional_encode:
              self.attention_mask = _attention_mask  # for decoder
            def cat_by_doc(raw_tensor, memory_bank_tensor):  # concat along the document dimension
              return torch.cat([
                raw_tensor.view(-1, self.n_passages, *raw_tensor.size()[1:]),
                memory_bank_tensor.view(-1, self.memory_bank_topk, *memory_bank_tensor.size()[1:])], dim=1).view(
                -1, *raw_tensor.size()[1:])
            # (bs * (n_context + memory_bank_topk), ...)
            input_ids = cat_by_doc(input_ids, _input_ids)
            attention_mask = cat_by_doc(attention_mask, _attention_mask)
            self.n_passages = self.n_passages + self.memory_bank_topk
          elif use_memory_bank:  # only store
            self.memory_bank.add_from_t5(
              query_states, key_states, attention_mask, input_ids, input_doc_ids, n_context=n_context)            
          if was_training:
            self.train()

        for block in self.encoder.block:
          block.n_passages = self.n_passages
          block.input_ids = input_ids
        
        if 'only_bi_encoder_forward' in kwargs and kwargs['only_bi_encoder_forward']:
          return self.bi_encoder_forward(input_ids, attention_mask)
        if 'only_bi_encoder_forward' in kwargs:
          del kwargs['only_bi_encoder_forward']

        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        if bool(self.encoder_encoder_kl_ratio):
          merge = self.get_collected_for_retrieval()
          # (num_query, n_context, num_layers, [num_heads, seq_len, seq_len])
          ttas = merge['two_tower_attn_score_full'] if 'two_tower_attn_score_full' in merge else merge['two_tower_attn_score']
          # (num_query, n_context, seq_len, seq_len)
          ttasfm = merge['two_tower_attn_score_full_mask'] if 'two_tower_attn_score_full' in merge else None
          self.compute_encoder_encoder_kl(ttas, ttasfm)
        if kwargs['return_dict']:
          outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, -1, outputs.last_hidden_state.size(-1))
        else:
          outputs = (outputs[0].view(bsz, -1, outputs[0].size(-1)),) + outputs[1:]
        return outputs

    def compute_encoder_encoder_kl(self,
                                   encoder_attn: torch.FloatTensor, # (num_query, n_context, num_layers, [num_heads, seq_len, seq_len])
                                   encoder_attn_mask: torch.BoolTensor):  # (num_query, n_context, seq_len, seq_len)
      layers, heads = self.encoder_encoder_kl.split('=')
      num_query, num_layers = encoder_attn.size(0), encoder_attn.size(2)
      if layers == 'first|last':
        assert num_layers == 2
        pred_attn = encoder_attn[:, :, 0]  # (num_query, n_context, num_heads, [seq_len, seq_len])
        gold_attn = encoder_attn[:, :, -1]  # (num_query, n_context, num_heads, [seq_len, seq_len])
      else:
        raise NotImplementedError
      pred_head, gold_head = heads.split('|')
      if pred_head.isnumeric() and gold_head.isnumeric():  # specified head
        pred_head, gold_head = int(pred_head), int(gold_head)
        pred_attn = pred_attn[:, :, pred_head]  # (num_query, n_context, [seq_len, seq_len])
        gold_attn = gold_attn[:, :, gold_head]  # (num_query, n_context, [seq_len, seq_len])
      else:
        raise NotImplementedError

      if self.encoder_encoder_kl_sparsity:  # choose topk doc token per query
        seq_len = pred_attn.size(2)
        pred_attn = pred_attn - (~encoder_attn_mask * 1e5)
        gold_attn = gold_attn - (~encoder_attn_mask * 1e5)
        topk = min(self.encoder_encoder_kl_sparsity, seq_len)
        pred_topk_ind = pred_attn.topk(topk, -1)[1]  # (num_query, n_context, seq_len, topk)
        gold_topk_ind = gold_attn.topk(topk, -1)[1]  # (num_query, n_context, seq_len, topk)
        sparsity_mask = torch.zeros_like(encoder_attn_mask).bool()  # (num_query, n_context, seq_len, seq_len)
        sparsity_mask.scatter_(-1, pred_topk_ind, True)
        sparsity_mask.scatter_(-1, gold_topk_ind, True)
        encoder_attn_mask = encoder_attn_mask & sparsity_mask
        WandbLogger.log_w_step({
          'encoder-kl-sparsity': (encoder_attn_mask.sum([2, 3]) / encoder_attn_mask.max(-1)[0].sum(-1)).mean().item()})

      pred_attn = pred_attn.contiguous().view(num_query, -1)  # (num_query, n_context * [seq_len * seq_len])
      gold_attn = gold_attn.contiguous().view(num_query, -1)  # (num_query, n_context * [seq_len * seq_len])

      if encoder_attn_mask is not None:
        encoder_attn_mask = encoder_attn_mask.view(num_query, -1)  # (num_query, n_context * seq_len * seq_len)
        pred_attn = pred_attn - (~encoder_attn_mask * 1e5)
        gold_attn = gold_attn - (~encoder_attn_mask * 1e5)

      pred_attn_logprob = torch.log_softmax(pred_attn, dim=-1)
      gold_attn_prob = torch.softmax(gold_attn, dim=-1)

      kl_loss_func = torch.nn.KLDivLoss(reduction='none')
      if encoder_attn_mask is not None:
        pred_attn_logprob = torch.masked_select(pred_attn_logprob, encoder_attn_mask)  # (?)
        gold_attn_prob = torch.masked_select(gold_attn_prob, encoder_attn_mask)  # (?)
      kl = kl_loss_func(pred_attn_logprob, gold_attn_prob.detach())  # (?)
      kl = kl.sum() / num_query  # bachmean
      self.kl_loss = kl

    def get_kl_loss(self):
      assert self.encoder_encoder_kl_ratio
      kl_loss = self.kl_loss * self.encoder_encoder_kl_ratio
      WandbLogger.log_w_step({'encoder-kl-loss': kl_loss.item()})
      return kl_loss

    def apply_t5block_wrapper(self, config):
      def _bi_encoder_forward(
           input_ids,  # (bs, seq_len)
           attention_mask):  # (bs, seq_len, seq_len)
        if self.memory_bank_additional_encode and self.training:
          self.attention_mask = attention_mask  # for decoder
        self.n_passages = self.n_passages + self.memory_bank_topk  # for encoder-decoder kl loss
        return self.bi_encoder_forward(input_ids, attention_mask)
      nl_total = len(self.encoder.block)
      nl_twotower = config.n_layer_two_tower
      use_first_layer = lambda i: i == nl_twotower
      use_emb_layer = lambda i: i == 0
      use_emb_and_first_layer = lambda i: i == 0 or i == nl_twotower
      use_prev_and_first_layers = lambda i: i <= nl_twotower
      use_after_and_first_layers = lambda i: i >= nl_twotower
      use_last_and_first_layers = lambda i: i == nl_twotower or i == nl_total - 1
      if config.layer_for_retrieval == 'first':
        use_for_retrieval_func = use_first_layer
      elif config.layer_for_retrieval == 'emb':
        use_for_retrieval_func = use_emb_layer
      elif config.layer_for_retrieval == 'emb-first':
        use_for_retrieval_func = use_emb_and_first_layer
      elif config.layer_for_retrieval == 'prev-first':
        use_for_retrieval_func = use_prev_and_first_layers
      elif config.layer_for_retrieval == 'after-first':
        use_for_retrieval_func = use_after_and_first_layers
      elif config.layer_for_retrieval == 'last-first':
        use_for_retrieval_func = use_last_and_first_layers
      else:
        raise NotImplementedError
      layers = []
      self.retrieval_t5block_funcs = []
      for i, layer in enumerate(self.encoder.block):
        use_for_retrieval = use_for_retrieval_func(i)
        wrapped_layer = T5blockWrapper(
          config,
          layer,
          layer_index=i,
          use_checkpoint=self.use_checkpoint,
          use_full_attention=i >= config.n_layer_two_tower,
          use_for_retrieval=use_for_retrieval,
          bi_encoder_forward=_bi_encoder_forward)
        if use_for_retrieval:
          self.retrieval_t5block_funcs.append(wrapped_layer.get_collected_for_retrieval)
        layers.append(wrapped_layer)
      self.encoder.block = nn.ModuleList(layers)

    def get_collected_for_retrieval(self):
      merge = {}
      for i, func in enumerate(self.retrieval_t5block_funcs):
        result = func()
        for k, v in result.items():  # v is (bs, num_heads, ...) or (bs, ?, num_heads, ...) for two_tower attn_score
          if k.endswith('mask'):  # mask is the same across layers
            merge[k] = v
            continue
          if k not in merge:
            merge[k] = [v]
          else:
            merge[k].append(v)
      for k in merge:
        if k.startswith('two_tower_attn_score'):
          merge[k] = torch.stack(merge[k], 2)  # (bs, ?, num_layers, num_heads, ...)
        elif not k.endswith('mask'):
          merge[k] = torch.stack(merge[k], 1)  # (bs, num_layers, num_heads, ...)
        if hasattr(self, 'n_passages') and not k.startswith('two_tower_attn_score'):
          # (num_query, n_context, num_layers, num_heads, ...)
          merge[k] = merge[k].view(*((-1, self.n_passages) + merge[k].size()[1:]))
      if hasattr(self, 'n_passages'):  # used by decoder
        merge['n_context'] = self.n_passages
      return merge
    
    def set_accumulated(self, accumulated: Dict):
      self.encoder.block[self.n_layer_two_tower].accumulated = accumulated
    
    def delete_accumulated(self):
      if hasattr(self.encoder.block[self.n_layer_two_tower], 'accumulated'):
        del self.encoder.block[self.n_layer_two_tower].accumulated

class T5blockWrapper(torch.nn.Module):
    """
    (1) replacing None outputs by empty tensors, which allows the use of checkpointing.
    (2) added code to handle separate/full attention at different layers.
    """
    def __init__(self,
                 config,
                 module,
                 layer_index: int,
                 use_checkpoint: bool = False,
                 use_full_attention: bool = True,
                 use_for_retrieval: bool = False,
                 bi_encoder_forward: Callable = None):
        super().__init__()
        self.module = module
        self.layer_index = layer_index
        self.use_checkpoint = use_checkpoint
        self.use_full_attention = use_full_attention
        self.use_for_retrieval = use_for_retrieval
        self.bi_encoder_forward = bi_encoder_forward
        self.retrieval_aggregation_method = config.retrieval_aggregation_method
        self.term_weight_parameter = config.term_weight_parameter
        self.embedding_normalize = config.embedding_normalize
        self.max_over_head = config.max_over_head
        self.in_batch_negative = config.in_batch_negative
        self.in_batch_negative_size = config.in_batch_negative_size
        self.in_batch_negative_max_num_query = config.in_batch_negative_max_num_query
        self.memory_bank = None
        self.memory_bank_topk = config.memory_bank_topk
        self.memory_use_random = config.memory_use_random
        self.memory_bank_recompute = config.memory_bank_recompute
        self.memory_bank_additional_encode = config.memory_bank_additional_encode
        self.pad_token_id = config.pad_token_id
        self.num_heads = config.num_heads
        self.n_layer_two_tower = config.n_layer_two_tower
        self.n_context_for_ibn = config.n_context_for_ibn
        self.need_collect = FiDT5.need_wrap_decoder(config) or bool(config.encoder_encoder_kl_ratio)
        self.retrieval_projection = config.retrieval_projection
        self.term_weight_linear = None
        if self.use_for_retrieval and self.need_collect and self.term_weight_parameter:
          self.term_weight_linear = nn.Linear(config.d_kv, 1, bias=True)
          nn.init.constant_(self.term_weight_linear.weight, 0.0)
          nn.init.constant_(self.term_weight_linear.bias, 0.0)
        self.hidden_linear = self.head_linear = None
        if self.use_for_retrieval and self.need_collect:
          if not self.retrieval_projection:
            pass
          elif self.retrieval_projection.startswith('hidden_linear.'):
            retrieval_dim = int(self.retrieval_projection[len('hidden_linear.'):])
            self.hidden_linear = nn.Linear(config.d_model, retrieval_dim, bias=False)
            self.hidden_linear.weight.data.normal_(mean=0.0, std=config.initializer_factor * (config.d_model ** -0.5))
          elif self.retrieval_projection.startswith('head_linear.'):
            retrieval_dim = int(self.retrieval_projection[len('head_linear.'):])
            self.head_linear = nn.Linear(config.d_kv, retrieval_dim, bias=False)
            self.head_linear.weight.data.normal_(mean=0.0, std=config.initializer_factor * (config.d_kv ** -0.5))
          else:
            raise NotImplementedError

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        # register callback for retrieval
        if self.use_for_retrieval and (not self.training or self.need_collect):
            reg_point = self.module.layer[0]
            reg_point.SelfAttention.collect_for_retrieval = types.MethodType(
                functools.partial(
                    collect_for_retrieval,
                    attention_mask=attention_mask[:, 0].eq(0),
                    input_ids=self.input_ids if hasattr(self, 'input_ids') else None,  # passed from the forward function of the encoder
                    aggregation_method=self.retrieval_aggregation_method,
                    max_over_head=self.max_over_head,
                    field='query',
                    use_hidden_states=False,
                    n_context=self.n_passages if hasattr(self, 'n_passages') else None,
                    use_head_idx=get_single_head_idx(self.num_heads, self.n_layer_two_tower, self.max_over_head, layer_index=self.layer_index),
                    term_weight_linear=self.term_weight_linear,
                    embedding_normalize=self.embedding_normalize,
                    hidden_linear=self.hidden_linear,
                    head_linear=self.head_linear,
                    n_context_for_ibn=self.n_context_for_ibn,
                    accumulated=self.accumulated if hasattr(self, 'accumulated') else None), reg_point.SelfAttention)
            if self.in_batch_negative and self.training:
              reg_point.SelfAttention.in_batch_negative = types.MethodType(functools.partial(
                in_batch_negative, n_context=self.n_passages, in_batch_negative_size=self.in_batch_negative_size,
                max_num_query_per_compute=self.in_batch_negative_max_num_query), reg_point.SelfAttention)
        # handle separate/full attention
        if self.use_full_attention:
            # modify (potentially separate) attention mask to make it a full attention
            attention_mask = attention_mask.max(2, keepdim=True)[0]  # (bs, num_heads, 1, seq_len)
        # handle checkpointing
        if self.use_checkpoint and self.training:
            # when memory bank requires recompute, do not use checkpoint for retrieval layer
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

            output = checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        if self.use_for_retrieval and not self.training:
            # make the reg dynamic (only use in eval since training use checkpointing)
            reg_point = self.module.layer[0].SelfAttention
            del reg_point.collect_for_retrieval
        return output

    def get_collected_for_retrieval(self):
        reg_point = self.module.layer[0].SelfAttention
        return reg_point.retrieval

def collect_cross_attention(
     self,
     scores: torch.FloatTensor,  # (bs, n_heads, 1, enc_seq_len) where masked positions have -inf
     mask: torch.BoolTensor,  # (bs, n_heads or 1, 1, enc_seq_len)
     n_context: int,
     sum_over_tokens: bool,
     use_softmax: bool = False):
  # apply softmax
  if use_softmax:
    scores = torch.softmax(scores.float(), dim=-1).type_as(scores)
  # save cross attn at all decoding position
  if sum_over_tokens:
    # TODO: we sum over text_len to save space but is there any better workaround?
    s = scores.view(*(scores.size()[:3] + (n_context, -1)))  # (bs, n_heads, 1, n_context, text_len)
    m = mask.view(*(mask.size()[:3] + (n_context, -1)))  # (bs, n_heads or 1, 1, n_context, text_len)
    s = (s * m).sum(-1)  # (bs, n_heads, 1, n_context)
  else:
    s = scores
  if self.score_storage is None:
    self.score_storage = [s]
  else:
    self.score_storage.append(s)

def decoder_attn_ctx_normalize(
     self,
     score: torch.FloatTensor,  # (bs, n_heads, dec_seq_len, enc_seq_len) where masked positions have -inf
     n_context: int):
  raw_size = score.size()
  s = score.view(*(raw_size[:3] + (n_context, -1)))  # (bs, n_heads, dec_seq_len, n_context, text_len)
  s = s - torch.logsumexp(s, -1, keepdim=True)
  s = s.view(*raw_size)  # (bs, n_heads, dec_seq_len, n_context * text_len)
  return s

def compute_kl_loss_reduction(
    prediction_logits: torch.FloatTensor,  # (bs, num_docs)
    gold_probs: torch.FloatTensor,  # (bs, num_docs)
    kl_loss_func: Callable,
    num_pos: int = None,
    num_neg: int = None,
    only_one_positive: bool = False):
  preds: List[torch.FloatTensor] = []
  golds: List[torch.FloatTensor] = []
  for b in range(gold_probs.size(0)):
    pos_doc_idxs = gold_probs[b].nonzero(as_tuple=False).squeeze(-1).sort().values
    neg_doc_idxs = (~gold_probs[b].bool()).nonzero(as_tuple=False).squeeze(-1).tolist()
    assert len(pos_doc_idxs) + len(neg_doc_idxs) == gold_probs.size(1)
    if only_one_positive:  # each pos doc corresponds to a separate softmax
      inf_mask = torch.zeros_like(prediction_logits[b])
      inf_mask.scatter_(0, pos_doc_idxs, 1e5)
      for doc_idx in pos_doc_idxs:
          _gold = torch.zeros_like(gold_probs[b])
          _gold[doc_idx] = 1.0  # one-hot
          _inf_mask = inf_mask.scatter(0, doc_idx, 0)
          if num_neg and len(neg_doc_idxs) > num_neg:  # use at most num_neg neg docs
            discard_neg_doc_idxs = random.sample(neg_doc_idxs, k=len(neg_doc_idxs) - num_neg)
            _inf_mask = _inf_mask.scatter(0, torch.tensor(discard_neg_doc_idxs).to(pos_doc_idxs), 1e5)
          _pred = prediction_logits[b] - _inf_mask
          golds.append(_gold)
          preds.append(_pred)
    else:
      inf_mask = torch.zeros_like(prediction_logits[b])
      _gold = gold_probs[b]
      if num_pos and len(pos_doc_idxs) > num_pos:  # use the top num_pos pos docs (assume pos docs are at the beginning)
        discard_pos_doc_idxs = pos_doc_idxs[num_pos:]
        inf_mask = inf_mask.scatter(0, discard_pos_doc_idxs, 1e5)
        _gold[num_pos:] = 0
      if num_neg and len(neg_doc_idxs) > num_neg:  # use at most num_neg neg docs
        discard_neg_doc_idxs = random.sample(neg_doc_idxs, k=len(neg_doc_idxs) - num_neg)
        inf_mask = inf_mask.scatter(0, torch.tensor(discard_neg_doc_idxs).to(pos_doc_idxs), 1e5)
      _pred = prediction_logits[b] - inf_mask
      golds.append(_gold / (_gold.sum() or 1))
      preds.append(_pred)
  if len(golds):
    preds = torch.stack(preds, dim=0)
    golds = torch.stack(golds, dim=0)
    preds = torch.log_softmax(preds, dim=-1)
    kl_loss = kl_loss_func(preds, golds)
  else:  # no annotation
    kl_loss = kl_loss_func(torch.log_softmax(prediction_logits, dim=-1), gold_probs)
  return kl_loss

def encoder_decoder_kl(
     self,
     decoder_score: torch.FloatTensor,  # (bs, n_heads, dec_seq_len, enc_seq_len)
     decoder_mask: torch.BoolTensor,  # (bs, n_heads or 1, dec_seq_len, enc_seq_len)
     encoder_score: torch.FloatTensor,  # (bs, n_context, [text_len]) or (<=(n_gpu * bs)^2, n_context) or (bs + ?, n_context) or (num_layer, bs, n_context)
     encoder_score_mask: torch.BoolTensor,  # (bs, n_context, text_len)
     n_context: int,
     only_topk_n_context: int = 0,
     use_softmax: bool = False,
     encoder_score_pre_softmaxed: bool = False,
     in_batch_negative: bool = False,
     pairwise_loss: str = None,
     memory_bank_topk: int = 0,
     memory_bank_additional_encode: bool = False,
     gold_doc_dist: torch.FloatTensor=None,  # (bs, n_context)
     use_gold_doc_dist: bool = False,
     kl_loss_reduction: str = None,
     encoder_decoder_kl_method: str = 'merge',
     encoder_encoder_kl_method: str = None,
     accumulated: Dict = None):

  if encoder_decoder_kl_method == 'merge':  # add the layer dimension
    encoder_score = encoder_score.unsqueeze(0)
    encoder_score_mask = encoder_score_mask.unsqueeze(0) if encoder_score_mask is not None else None
  num_layer = encoder_score.size(0)
  multi_layer_log_w_step = lambda log_key, value: WandbLogger.log_w_step({log_key: value}, prefix_for_list='L', skip_first_for_list=True)

  bs, _, _, enc_seq_len = decoder_score.size()
  has_token = encoder_score.dim() == 4
  use_memory_bank = memory_bank_topk and \
                    not memory_bank_additional_encode and \
                    encoder_score.size(1) > bs  # the first several batch might not use memory-bank

  if only_topk_n_context:  # only consider the topk context to mimic in-batch negative
    only_topk_enc_tok = (enc_seq_len // n_context) * only_topk_n_context
    decoder_score[:, :, :, only_topk_enc_tok:] = -1e5

  # apply softmax
  if use_softmax:
    decoder_score = torch.softmax(decoder_score.float(), dim=-1).type_as(decoder_score)
  # avg over head, use the first decoder tok
  decoder_score = decoder_score.mean(1)[:, 0]  # (bs, enc_seq_len)
  decoder_mask = decoder_mask[:, 0, 0]  # (bs, enc_seq_len)

  if has_token:  # token-level kl
    assert num_layer == 1
    if in_batch_negative:
      raise NotImplementedError
    assert not encoder_score_pre_softmaxed
    encoder_score = encoder_score.view(bs, -1)  # (bs, enc_seq_len)
    encoder_score_mask = encoder_score_mask.view(bs, -1)  # (bs, enc_seq_len)
    dec_attn = decoder_score if use_softmax else torch.softmax(decoder_score, dim=-1)
    kl_loss_func = torch.nn.KLDivLoss(reduction='none')
    dec_attn = torch.masked_select(dec_attn, encoder_score_mask)  # (?)
    dec_attn = dec_attn / (dec_attn.sum() + 1e-10)  # redistribute
    enc_attn = torch.masked_select(encoder_score, encoder_score_mask)  # (?) softmax is already applied
    kl = kl_loss_func(enc_attn, dec_attn.detach())  # (?)
    kl = kl.sum() / bs  # bachmean
    return kl

  # sum over doc tokens
  s = decoder_score.view(bs, n_context, -1)  # (bs, n_context, text_len)
  m = decoder_mask.view(bs, n_context, -1)  # (bs, n_context, text_len)
  s = (s * m).sum(-1)  # (bs, n_context)  # TODO: avg instead of sum since avg is used in final evaluation?

  # use the gold distribution
  if use_gold_doc_dist and gold_doc_dist is not None:
    if kl_loss_reduction is None:
      s = gold_doc_dist / (gold_doc_dist.sum(-1, keepdim=True) + 1e-5)  # sum to one
    else:
      s = gold_doc_dist

  # kl
  kldiv = torch.nn.KLDivLoss(reduction='batchmean')
  def kl_loss_func(pred, gold, log_key: str = 'kl-loss'):
    assert gold.dim() == 2
    if pred.dim() == 2:
      loss = kldiv(pred, gold)
      WandbLogger.log_w_step({log_key: loss.item()})
      return loss
    if pred.dim() == 3:  # multiple layers
      losses = [kldiv(_pred, gold) for _pred in pred]
      multi_layer_log_w_step(log_key, losses)
      loss = 0
      for _loss in losses:
        loss += _loss
      return loss
    raise Exception(f'prediction distribution shape {pred.size()} incorrect')
  
  if use_softmax:
    dec_attn = s
  else:
    dec_attn = torch.softmax(s, dim=-1)  # no grad to decoder
  self.retrieval = {'decoder_attention': dec_attn}
  
  if in_batch_negative:  # collect dec_attn across gpus
    if accumulated is not None:
      real_rank = accumulated['step'] * get_world_size() + get_rank()
      _dec_attn = list(accumulated['decoder_attention'])
      _dec_attn[real_rank] = dec_attn
      dec_attn = torch.cat(_dec_attn, dim=0).to(dec_attn).detach()  # (n_gpu * bs * total, n_context)
    else:
      dec_attn, = all_gather_tensors(dec_attn)
      dec_attn = torch.cat(dec_attn, dim=0).to(dec_attn[0]).detach()  # (n_gpu * bs, n_context)
  #WandbLogger.log_w_step({'decoder-dist-kl': torch.sort(dec_attn[0], descending=True)[0][:10]})
  if encoder_score_pre_softmaxed:
    if in_batch_negative:
      raise NotImplementedError
    enc_attn = encoder_score  # already with log
  else:
    if in_batch_negative:
      bs_t_ngpu = dec_attn.size(0)
      encoder_score = encoder_score.view(num_layer, bs_t_ngpu, -1)  # (num_layer, n_gpu * bs, <=(n_gpu * bs) * n_context)
      if pairwise_loss is None:  # kl over all docs
        enc_attn = torch.log_softmax(encoder_score, dim=-1)
      elif pairwise_loss == 'sigmoid':  # kl over current docs, sigmoid over others
        enc_attn = torch.log_softmax(encoder_score[:, :, :n_context], dim=-1)
      else:
        raise NotImplementedError
    elif use_memory_bank:
      encoder_score = encoder_score.view(num_layer, -1, n_context + memory_bank_topk)  # (num_layer, bs, n_context + memory_bank_topk)
      enc_attn = torch.log_softmax(encoder_score, dim=-1)
    else:
      if kl_loss_reduction is not None:
        enc_attn = encoder_score  # no need to normalize
      else:
        enc_attn = torch.log_softmax(encoder_score, dim=-1)
  
  if encoder_decoder_kl_method == 'cross':  # assue the last layer is the cross encoder layer
    enc_attn_for_ed_kl = enc_attn[-1:]
    encoder_score_for_ed_kl = encoder_score[-1:]
  else:
    enc_attn_for_ed_kl = enc_attn
    encoder_score_for_ed_kl = encoder_score

  if in_batch_negative:
    if pairwise_loss is None:  # kl over all docs
      # (n_gpu * bs, <=(n_gpu * bs) * n_context)
      dec_attn = torch.cat([dec_attn, torch.zeros((bs_t_ngpu, enc_attn_for_ed_kl.size(2) - n_context)).to(enc_attn_for_ed_kl)], dim=-1)
      loss = kl_loss_func(enc_attn_for_ed_kl, dec_attn.detach())
      pos_prob_sum = enc_attn_for_ed_kl[:, :, :n_context].exp().sum(-1).mean(-1)  # (num_layer,)
      multi_layer_log_w_step('current-docs-encoder-prob-sum', pos_prob_sum)
      ori_kl = kl_loss_func(torch.log_softmax(encoder_score_for_ed_kl[:, :, :n_context], dim=-1), dec_attn[:, :n_context].detach(), log_key='kl-loss-original')
      if memory_bank_topk and memory_bank_additional_encode:  # track the original kl when additional_encode is used
        ori_n_context = n_context - memory_bank_topk
        pos_prob_sum = enc_attn_for_ed_kl[:, :, :ori_n_context].exp().sum(-1).mean(-1)  # (num_layer,)
        multi_layer_log_w_step('current-docs-encoder-prob-sum2', pos_prob_sum)
        ori_kl = kl_loss_func(
          torch.log_softmax(encoder_score_for_ed_kl[:, :, :ori_n_context], dim=-1),
          (dec_attn[:, :ori_n_context] / (dec_attn[:, :ori_n_context].sum(-1, keepdim=True) + 1e-10)).detach(), log_key='kl-loss-original2')
    elif pairwise_loss == 'sigmoid':  # kl over current docs, sigmoid over others  # TODO: not tested
      raise NotImplementedError
      loss = kl_loss_func(enc_attn, dec_attn.detach())
      WandbLogger.log_w_step({'kl-loss': loss.item()})
      # (n_gpu * bs, n_context, <=(n_gpu * bs - 1) * n_context)
      margin = encoder_score[:, :n_context].unsqueeze(-1) - encoder_score[:, n_context:].unsqueeze(1)
      margin_loss = -F.logsigmoid(margin).mean(-1).sum(-1).mean(0)
      WandbLogger.log_w_step({'sigmoid-loss': margin_loss.item()})
      loss = loss + margin_loss
    else:
      raise NotImplementedError
  elif use_memory_bank:
    raise NotImplementedError
    dec_attn = torch.cat([dec_attn, torch.zeros((bs, memory_bank_topk)).to(dec_attn)], dim=-1)  # (bs, n_context + memory_bank_topk)
    loss = kl_loss_func(enc_attn, dec_attn.detach())
    WandbLogger.log_w_step({'kl-loss': loss.item()})
    pos_prob_sum = enc_attn[:, :n_context].exp().sum(-1).mean().item()
    WandbLogger.log_w_step({'current-docs-encoder-prob-sum': pos_prob_sum})
    ori_kl = kl_loss_func(torch.log_softmax(encoder_score[:, :n_context], dim=-1), dec_attn[:, :n_context].detach())
    WandbLogger.log_w_step({'kl-loss-original': ori_kl.item()})
  else:
    if kl_loss_reduction == 'only_one_positive':
      loss = compute_kl_loss_reduction(enc_attn_for_ed_kl, dec_attn.detach(), kl_loss_func=kl_loss_func, only_one_positive=True)
    elif kl_loss_reduction == 'only_one_positive-neg3':
      loss = compute_kl_loss_reduction(enc_attn_for_ed_kl, dec_attn.detach(), kl_loss_func=kl_loss_func, num_neg=3, only_one_positive=True)
    elif kl_loss_reduction == 'neg3':
      loss = compute_kl_loss_reduction(enc_attn_for_ed_kl, dec_attn.detach(), kl_loss_func=kl_loss_func, num_neg=3, only_one_positive=False)
    elif kl_loss_reduction == 'pos1-neg3':
      loss = compute_kl_loss_reduction(enc_attn_for_ed_kl, dec_attn.detach(), kl_loss_func=kl_loss_func, num_pos=1, num_neg=3, only_one_positive=False)
    elif kl_loss_reduction == 'pos1':
      loss = compute_kl_loss_reduction(enc_attn_for_ed_kl, dec_attn.detach(), kl_loss_func=kl_loss_func, num_pos=1, only_one_positive=False)
    else:
      loss = kl_loss_func(enc_attn_for_ed_kl, dec_attn.detach())
    if memory_bank_topk and memory_bank_additional_encode:  # track the original kl when additional_encode is used
      ori_n_context = n_context - memory_bank_topk
      pos_prob_sum = enc_attn_for_ed_kl[:, :, :ori_n_context].exp().sum(-1).mean(-1)  # (num_layer)
      multi_layer_log_w_step('current-docs-encoder-prob-sum', pos_prob_sum)
      ori_kl = kl_loss_func(
        torch.log_softmax(encoder_score_for_ed_kl[:, :, :ori_n_context], dim=-1),
        (dec_attn[:, :ori_n_context] / (dec_attn[:, :ori_n_context].sum(-1, keepdim=True) + 1e-10)).detach(), log_key='kl-loss-original')
  
  # encoder kl
  if encoder_encoder_kl_method is None:
    pass
  elif encoder_encoder_kl_method == 'side':  # use the top and bottom layer
    enc_kl_loss = kl_loss_func(enc_attn[0], enc_attn[-1].detach().exp(), log_key='encoder-kl-loss')
    loss += enc_kl_loss
  else:
    raise NotImplementedError
 
  return loss

def aggregate_attention(
     self,
     scores: torch.FloatTensor,  # (bs, n_heads, seq_len, seq_len)
     attention_mask: torch.BoolTensor,  # (bs, seq_len, seq_len)
     field: str,
     aggregation_method: str,
     max_over_head: bool = False,
     use_head_idx: int = None,
     n_heads: int = None,
     term_weights: torch.FloatTensor = None):  # (bs, n_heads, seq_len)
  pad_mask = attention_mask.max(1)[0]  # (bs, seq_len)
  query_field_mask = attention_mask[:, 0]  # (bs, seq_len)
  doc_field_mask = attention_mask[:, -1]  # (bs, seq_len)
  if field == 'query':
    field_mask = query_field_mask
  elif field == 'doc':
    field_mask = doc_field_mask
  elif field == 'all':
    field_mask = pad_mask
  else:
    raise NotImplementedError
  cross_mask = (~attention_mask & pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)).unsqueeze(
    1)  # (bs, 1, seq_len, seq_len)

  if max_over_head:
    scores = scores.max(1, keepdim=True)[0]

  if aggregation_method == 'all-max-all':
    # max over tokens paying attention
    scores_full = (scores - (~cross_mask * 1e5)).max(2)[0]  # (bs, n_heads, seq_len)
    scores = (scores_full.exp() * doc_field_mask.unsqueeze(1)).sum(
      -1)  # (bs, n_heads)  TODO: exp not numerically stable
    self.retrieval = {'two_tower_attn_score': scores,
                      'two_tower_attn_score_full': scores_full,
                      'two_tower_attn_score_full_mask': doc_field_mask.contiguous()}
    return

  if aggregation_method == 'all-all-all':
    scores_full = scores * cross_mask  # (bs, n_heads, seq_len, seq_len)
    self.retrieval = {'two_tower_attn_score_full': scores_full,
                      'two_tower_attn_score_full_mask': cross_mask[:, 0] & field_mask[:, :, None]}  # (bs, seq_len, seq_len)
    aggregation_method = 'all-avg-max'

  head_agg, query_agg, key_agg = aggregation_method.split('-')
  if key_agg == 'max':
    scores = (scores - (~cross_mask * 1e5)).max(-1)[0]  # (bs, n_heads, seq_len)
  elif key_agg == 'avg':
    scores = (scores * cross_mask).sum(-1) / (cross_mask.sum(-1) + 1e-10)  # (bs, n_heads, seq_len)
  elif key_agg == 'maxsp':
    cross_mask = cross_mask.repeat(1, scores.size(1), 1, 1)
    max_sparsify(scores, cross_mask, -1, inplace=True)  # (bs, n_heads, seq_len, seq_len)
  else:
    raise NotImplementedError

  if query_agg == 'avg':
    if term_weights is not None:
      scores = (scores * term_weights).sum(-1)
    else:
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

  if use_head_idx is not None:
    scores = torch.cat([torch.zeros_like(scores).repeat(1, use_head_idx), scores,
                        torch.zeros_like(scores).repeat(1, n_heads - use_head_idx - 1)], dim=1)
  self.retrieval['two_tower_attn_score'] = scores

def get_single_head_idx(num_heads: int, n_layer_two_tower: int, max_over_head: bool = False, layer_index: int = None):
  if layer_index is None:
    layer_index = n_layer_two_tower
  if max_over_head:
    return None
  if num_heads == 12:  # 'google/t5-base-lm-adapt':
    assert n_layer_two_tower == 6
    if layer_index == 6:
      return 3
    if layer_index == 11:
      return 2
    raise NotImplementedError
  if num_heads == 16:  # 'google/t5-large-lm-adapt':
    assert n_layer_two_tower in {6, 12, 18}
    if layer_index == 6:
      return 5
    if layer_index == 12:
      return 6
    if layer_index == 18:
      return 11
    raise NotImplementedError
  raise NotImplementedError

def compute_score_and_aggregate_in_batch(
  query_emb: torch.FloatTensor,  # (nq1, n_context, n_heads, max_qry_len, emb_size_per_head)
  doc_emb: torch.FloatTensor,  # (nq2, n_context, n_heads, max_doc_len, emb_size_per_head)
  query_mask: torch.BoolTensor,  # (nq1, n_context, max_qry_len)
  doc_mask: torch.BoolTensor,  # (nq2, n_context, max_doc_len)
  position_bias: torch.FloatTensor,  # (n_heads, max_qry_len, max_doc_len)
  aggregation_method: str = 'all-avg-max',
  batch_size: int = None,
  no_grad: bool = False,
  iteration: str = 'cross'):
  assert iteration in {'self', 'cross'}
  nq1 = query_emb.size(0)
  nq2 = doc_emb.size(0)
  bsz1 = batch_size or nq1
  bsz2 = batch_size or nq2
  query_embs = query_emb.split(bsz1)
  query_masks = query_mask.split(bsz1)
  doc_embs = doc_emb.split(bsz2)
  doc_masks = doc_mask.split(bsz2)

  with torch.no_grad() if no_grad else contextlib.nullcontext():
    if iteration == 'cross':
      scores: List[List] = []
      for qe, qm in zip(query_embs, query_masks):
        scores.append([])
        for de, dm in zip(doc_embs, doc_masks):
          # (<=nq1, <=nq2, n_context, n_heads, max_qry_len, max_doc_len)
          s = torch.matmul(qe.unsqueeze(1), de.transpose(4, 3).unsqueeze(0))
          # (<=nq1, <=nq2, n_context, max_qry_len, max_doc_len)
          m = qm.unsqueeze(-1).unsqueeze(1) & dm.unsqueeze(-2).unsqueeze(0)
          s = s + position_bias[None, None, None]
          # aggregate scores
          assert aggregation_method == 'all-avg-max'
          # (<=nq1, <=nq2, n_context, n_heads, max_qry_len)
          s, max_doc_tok_idx = (s - (~m.unsqueeze(3) * 1e5)).max(-1)
          _qm = qm[:, None, :, None]  # (<=nq1, 1, n_context, 1, max_qry_len)
          # (<=nq1, <=nq2, n_context, n_heads)
          s = (s * _qm).sum(-1) / _qm.sum(-1)
          scores[-1].append(s)
        scores[-1] = torch.cat(scores[-1], dim=1)  # (<=nq1, nq2, n_context, n_heads)
      scores = torch.cat(scores, dim=0)  # (nq1, nq2, n_context, n_heads)
      assert scores.size(0) == nq1 and scores.size(1) == nq2
    elif iteration == 'self':
      assert nq1 == nq2
      scores: List = []
      for qe, qm, de, dm in zip(query_embs, query_masks, doc_embs, doc_masks):
        # (<=nq1, n_context, n_heads, max_qry_len, max_doc_len)
        s = torch.matmul(qe, de.transpose(4, 3))
        # (<=nq1, n_context, max_qry_len, max_doc_len)
        m = qm.unsqueeze(-1) & dm.unsqueeze(-2)
        s = s + position_bias[None, None]
        # aggregate scores
        assert aggregation_method == 'all-avg-max'
        # (<=nq1, n_context, n_heads, max_qry_len)
        s, max_doc_tok_idx = (s - (~m.unsqueeze(2) * 1e5)).max(-1)
        _qm = qm[:, :, None]  # (<=nq1, n_context, 1, max_qry_len)
        # (<=nq1, n_context, n_heads)
        s = (s * _qm).sum(-1) / _qm.sum(-1)
        scores.append(s)
      scores = torch.cat(scores, dim=0)  # (nq1, n_context, n_heads)
      assert scores.size(0) == nq1
    else:
      raise ValueError
    return scores

def collect_for_retrieval(
     self,
     scores: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, seq_len)
     hidden_states: torch.FloatTensor,  # (bs * n_context, seq_len, emb_size)
     query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
     key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
     value_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
     position_bias: torch.FloatTensor,  # (1, n_heads, seq_len, seq_len)
     preprocessed_mask: torch.FloatTensor,  # (bs * n_context, 1 (n_heads), seq_len, seq_len)
     attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
     input_ids: torch.LongTensor,  # (bs * n_context, seq_len)
     aggregation_method: str,  # 'head-query-key'
     max_over_head: bool,
     field: str,
     use_hidden_states: bool,
     n_context: int,
     memory_bank: MemoryBank = None,
     memory_bank_topk: int = 0,
     memory_use_random: bool = False,
     memory_bank_recompute: bool = False,
     memory_bank_additional_encode: bool = False,
     pad_token_id: int = 0,
     bi_encoder_forward: Callable = None,
     use_head_idx: int = None,
     term_weight_linear: nn.Linear = None,
     embedding_normalize: bool = False,
     hidden_linear: nn.Linear = None,
     head_linear: nn.Linear = None,
     n_context_for_ibn: int = None,
     accumulated: Dict = None):
  
  def _get_term_weights(query_states, query_mask):  # (?, n_heads, seq_len, emb_size_per_head), (?, seq_len)
    if term_weight_linear is None:
      return None
    weights = term_weight_linear(query_states)[:, :, :, 0]  # (?, n_heads, seq_len)
    weights = torch.softmax(weights / 0.01 - (~query_mask.unsqueeze(1) * 1e5), dim=-1)  # (?, n_heads, seq_len)
    #weights = weights * query_mask.unsqueeze(1)
    return weights
  
  if hidden_linear is not None:
    ret_states = hidden_linear(hidden_states)  # (bs * n_context, seq_len, ret_size)
    scores = torch.matmul(ret_states, ret_states.transpose(1, 2)).unsqueeze(1)  # (bs * n_context, 1, seq_len, seq_len)
    query_states = ret_states.unsqueeze(1)
    key_states = query_states
  elif head_linear is not None:
    query_states = head_linear(query_states)  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    key_states = head_linear(key_states)  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
    scores = torch.matmul(query_states, key_states.transpose(3, 2))  # (bs * n_context, n_heads, seq_len, seq_len)

  self.retrieval = {
    'scores': scores,
    'query_states': F.normalize(query_states, dim=-1, p=2) if embedding_normalize else query_states,
    'key_states': F.normalize(key_states, dim=-1, p=2) if embedding_normalize else key_states,
    'attention_mask': attention_mask,
    'value_states': value_states,
    'preprocessed_mask': preprocessed_mask}
  term_weights = _get_term_weights(query_states, attention_mask[:, 0])
  if term_weights is not None:
    self.retrieval['term_weights'] = term_weights
  if hasattr(self, 'just_collect') and self.just_collect:  # collect some tensors
    return

  n_heads = key_states.size(1)

  if hasattr(self, 'in_batch_negative') and self.training:  # collect and combine from all gpu (only in training)
    only_self_attn_qd = True  # only compute self attn between query query vectors and doc key vectors to save memory
    
    if only_self_attn_qd:
      nqpg = query_states.size(0) // n_context  # num query per gpu
      if accumulated is not None:  # use accumulated
        real_rank = accumulated['step'] * get_world_size() + get_rank()
        assert use_head_idx is not None
        global_q, global_k, global_attn = list(accumulated['query_states']), list(accumulated['key_states']), list(accumulated['attention_mask'])
        global_q[real_rank] = query_states[:, use_head_idx:use_head_idx + 1].contiguous()
        global_k[real_rank] = key_states[:, use_head_idx:use_head_idx + 1].contiguous()
        global_attn[real_rank] = attention_mask
      else:  # collect query, key, and attn across all gpus
        real_rank = get_rank()
        if use_head_idx is not None:
          global_q, global_k, global_attn = all_gather_tensors(
            query_states[:, use_head_idx:use_head_idx + 1].contiguous(), key_states[:, use_head_idx:use_head_idx + 1].contiguous(), attention_mask)
        else:
          global_q, global_k, global_attn = all_gather_tensors(query_states, key_states, attention_mask)
      # (n_q, n_context, n_heads, seq_len, emb_size_per_head) where n_q = n_gpu * bs [* accumulate_steps]
      global_q = torch.cat(global_q, dim=0).view(-1, n_context, *global_q[0].size()[1:])
      global_k = torch.cat(global_k, dim=0).view(-1, n_context, *global_k[0].size()[1:])
      # (n_q, n_context, seq_len, seq_len)
      global_attn = torch.cat(global_attn, dim=0).view(-1, n_context, *global_attn[0].size()[1:])
      nq = global_attn.size(0)
      start_index, end_index = nqpg * real_rank, nqpg * (real_rank + 1)

      # split into qry and doc representations
      max_qry_len = global_attn[:, :, 0].sum(-1).max()
      min_qry_len = global_attn[:, :, 0].sum(-1).min()
      query_mask = global_attn[:, :, 0, :max_qry_len]  # (n_q, n_context, max_qry_len)
      doc_mask = global_attn[:, :, -1, min_qry_len:]  # (n_q, n_context, seq_len - min_qry_len)
      global_q = global_q[:, :, :, :max_qry_len]  # (n_q, n_context, n_heads, max_qry_len, emb_size_per_head)
      global_k = global_k[:, :, :, min_qry_len:]  # (n_q, n_context, n_heads, seq_len - min_qry_len, emb_size_per_head)

      # crop position_bias
      if use_head_idx is not None:
        position_bias = position_bias[0, use_head_idx:use_head_idx + 1, :max_qry_len, min_qry_len:]  # (1, max_qry_len, seq_len - min_qry_len)
      else:
        position_bias = position_bias[0, :, :max_qry_len, min_qry_len:]  # (n_heads, max_qry_len, seq_len - min_qry_len)

      # TODO: term weight, normalization, max over head

      # compute self scores
      # (n_q, n_context, n_heads)
      scores = compute_score_and_aggregate_in_batch(
        global_q, global_k, query_mask, doc_mask, position_bias=position_bias, 
        aggregation_method=aggregation_method, no_grad=False, iteration='self')

      # sample a subset for the next step (TODO: bug)
      if n_context_for_ibn:
        assert n_context_for_ibn <= n_context
        # (n_q, n_context_for_ibn)
        sub_ind = torch.tensor([np.random.choice(n_context, size=n_context_for_ibn, replace=False) for _ in range(nq)]).to(attention_mask.device)
        sub_mask = torch.zeros(nq, n_context).bool().to(attention_mask.device)  # (n_q, n_context)
        sub_mask.scatter_(-1, sub_ind, True)
        # (n_q, n_context_for_ibn, n_heads, max_qry_len, emb_size_per_head)
        global_q_sub = torch.masked_select(global_q, sub_mask[:, :, None, None, None]).view(nq, n_context_for_ibn, *global_q.size()[2:])
        # (n_q, n_context_for_ibn, n_heads, seq_len - min_qry_len, emb_size_per_head)
        global_k_sub = torch.masked_select(global_k, sub_mask[:, :, None, None, None]).view(nq, n_context_for_ibn, *global_k.size()[2:])
        # (n_q, n_context_for_ibn, max_qry_len)
        query_mask_sub = torch.masked_select(query_mask, sub_mask[:, :, None]).view(nq, n_context_for_ibn, *query_mask.size()[2:])
        # (n_q, n_context_for_ibn, seq_len - min_qry_len)
        doc_mask_sub = torch.masked_select(doc_mask, sub_mask[:, :, None]).view(nq, n_context_for_ibn, *doc_mask.size()[2:])
      else:
        global_q_sub, global_k_sub, query_mask_sub, doc_mask_sub = global_q, global_k, query_mask, doc_mask

      # compute all ibn scores w/o grad
      # (n_q, n_q, <=n_context, n_heads)
      scores_cross = compute_score_and_aggregate_in_batch(
        global_q_sub, global_k_sub, query_mask_sub, doc_mask_sub, position_bias=position_bias, 
        aggregation_method=aggregation_method, batch_size=16, no_grad=True, iteration='cross')

      # compute local ibn scores w/ grad
      # (nqpg, n_q, <=n_context, n_heads)
      scores_cross_q = compute_score_and_aggregate_in_batch(
        global_q_sub[start_index:end_index], global_k_sub, query_mask_sub[start_index:end_index], doc_mask_sub, position_bias=position_bias, 
        aggregation_method=aggregation_method, no_grad=False, iteration='cross')  # TODO: add batch_size?
      # (n_q, nqpg, <=n_context, n_heads)
      scores_cross_d = compute_score_and_aggregate_in_batch(
        global_q_sub, global_k_sub[start_index:end_index], query_mask_sub, doc_mask_sub[start_index:end_index], position_bias=position_bias, 
        aggregation_method=aggregation_method, no_grad=False, iteration='cross')  # TODO: add batch_size?

      # merge ibn scores w/ and w/o grad
      scores_cross = torch.cat([scores_cross[:start_index], scores_cross_q, scores_cross[end_index:]], dim=0)
      scores_cross = torch.cat([scores_cross[:, :start_index], scores_cross_d, scores_cross[:, end_index:]], dim=1)

      # remove attn between qry and associated docs
      rm_mask = ~torch.eye(nq).bool().to(attention_mask.device)  # (n_q, n_q)
      gather_index = torch.masked_select(torch.arange(nq).unsqueeze(0).to(attention_mask.device), rm_mask).view(nq, nq - 1)  # (n_q, n_q - 1)
      # (n_q, (n_q - 1 * <=n_context), n_heads)
      scores_cross = torch.gather(scores_cross, 1, gather_index[:, :, None, None].repeat(1, 1, *scores_cross.size()[2:])).view(nq, -1, *scores_cross.size()[3:])

      # combine self scores and cross scores
      scores = torch.cat([scores, scores_cross], dim=1)  # (n_q, n_context + (nq - 1) * <=n_context, n_heads)
      if use_head_idx is not None:
        scores = torch.cat([
          torch.zeros_like(scores).repeat(1, 1, use_head_idx), 
          scores, 
          torch.zeros_like(scores).repeat(1, 1, n_heads - use_head_idx - 1)], dim=-1)
      self.retrieval['two_tower_attn_score'] = scores
    
    '''
    scores_li = []
    # (<=(n_gpu * bs)^2 * n_context, n_heads, seq_len, emb_size_per_head)
    # (<=(n_gpu * bs)^2 * n_context, seq_len, seq_len)
    for _query_states, _key_states, _attention_mask in self.in_batch_negative(
      query_states, key_states, attention_mask, head_idx=use_head_idx):

      if only_self_attn_qd:
        max_qry_len = _attention_mask[:, 0].sum(-1).max()  # (<=(n_gpu * bs)^2 * n_context)
        min_qry_len = _attention_mask[:, 0].sum(-1).min()  # (<=(n_gpu * bs)^2 * n_context)
        query_mask = _attention_mask[:, 0, :max_qry_len]  # (<=(n_gpu * bs)^2 * n_context, max_qry_len)
        doc_mask = _attention_mask[:, -1, min_qry_len:]  # (<=(n_gpu * bs)^2 * n_context, seq_len - min_qry_len)
        _query_states = _query_states[:, :, :max_qry_len]  # (<=(n_gpu * bs)^2 * n_context, n_heads, max_qry_len, emb_size_per_head)
        _key_states = _key_states[:, :, min_qry_len:]  # (<=(n_gpu * bs)^2 * n_context, n_heads, seq_len - min_qry_len, emb_size_per_head)
        if term_weight_linear is not None:
          term_weights = _get_term_weights(_query_states, query_mask)  # (<=(n_gpu * bs)^2 * n_context, n_heads, max_qry_len)
          WandbLogger.log_w_step({'term-weights': torch.sort(term_weights[0, 0], descending=True)[0][:10]})
        if embedding_normalize:
          _query_states = F.normalize(_query_states, dim=-1, p=2)
          _key_states = F.normalize(_key_states, dim=-1, p=2)
        # (<=(n_gpu * bs)^2 * n_context, n_heads, max_qry_len, seq_len - min_qry_len)
        scores = torch.matmul(_query_states, _key_states.transpose(3, 2))
        if use_head_idx is not None:
          scores = scores + position_bias[:, use_head_idx:use_head_idx + 1, :max_qry_len, min_qry_len:]
        else:
          scores = scores + position_bias[:, :, :max_qry_len, min_qry_len:]
        if term_weight_linear is not None:
          scores = (scores * term_weights.unsqueeze(-1))
        if max_over_head:
          assert use_head_idx is None
          scores, max_head_idx = scores.max(1, keepdim=True)  
        assert aggregation_method == 'all-avg-max'
        # (<=(n_gpu * bs)^2 * n_context, 1 (n_heads), max_qry_len, seq_len - min_qry_len)
        cross_mask = (query_mask.unsqueeze(-1) & doc_mask.unsqueeze(1)).unsqueeze(1)
        scores, max_doc_tok_idx = (scores - (~cross_mask * 1e5)).max(-1)  # (<=(n_gpu * bs)^2 * n_context, n_heads, max_qry_len)
        if max_over_head:
          max_head_idx = torch.gather(max_head_idx, -1, max_doc_tok_idx.unsqueeze(-1))  # (<=(n_gpu * bs)^2 * n_context, 1, max_qry_len, 1)
          max_head_idx, max_head_count = torch.unique_consecutive(max_head_idx.view(-1).sort().values, return_counts=True)  # (<=n_heads,)
          WandbLogger.log_w_step({'max-over-head': (max_head_idx, max_head_count)})
        if term_weight_linear is not None:
          scores = (scores * query_mask.unsqueeze(1)).sum(-1)
        else:
          scores = (scores * query_mask.unsqueeze(1)).sum(-1) / query_mask.unsqueeze(1).sum(-1)  # (<=(n_gpu * bs)^2 * n_context, n_heads)
        if use_head_idx is not None:
          scores = torch.cat([torch.zeros_like(scores).repeat(1, use_head_idx), scores,
                              torch.zeros_like(scores).repeat(1, n_heads - use_head_idx - 1)], dim=1)
        scores_li.append(scores)
      
      else:
        # (<=(n_gpu * bs)^2 * n_context, n_heads, seq_len, seq_len)
        scores = torch.matmul(_query_states, _key_states.transpose(3, 2))
        if use_head_idx is not None:
          scores = scores + position_bias[:, use_head_idx:use_head_idx + 1]
        else:
          scores = scores + position_bias
        aggregate_attention(
          self, scores, _attention_mask,
          field=field, aggregation_method=aggregation_method, max_over_head=max_over_head, 
          use_head_idx=use_head_idx, n_heads=n_heads, term_weights=_get_term_weights(_query_states, _attention_mask[:, 0]))
        scores_li.append(self.retrieval['two_tower_attn_score'])
      # ((n_gpu * bs)^2, n_context, n_heads, seq_len, seq_len)
      self.retrieval['two_tower_attn_score'] = torch.cat(scores_li, dim=0).view(-1, n_context, *scores_li[0].size()[1:])
    '''
    
    return

  if memory_bank is not None:
    raise NotImplementedError
    if term_weight_linear is not None:
      raise NotImplementedError
    assert use_head_idx is not None  # TODO: add multi-head?
    n_heads, seq_len, emb_size = key_states.size()[1:]
    # (bs, n_context, seq_len, emb_size_per_head)
    _key_states = key_states.view(-1, n_context, n_heads, seq_len, emb_size)[:, :, use_head_idx]
    _query_states = query_states.view(-1, n_context, n_heads, seq_len, emb_size)[:, :, use_head_idx]
    # (bs, n_context, seq_len, seq_len)
    _attention_mask = attention_mask.view(-1, n_context, seq_len, seq_len)
    # (bs, n_context, seq_len)
    _input_ids = input_ids.view(-1, n_context, seq_len)
    query_len = _attention_mask[:, 0, 0].sum(-1)  # (bs)

    need_retrieve = memory_bank_topk and len(memory_bank) >= memory_bank_topk
    if need_retrieve:  # query memory bank when there are enough
      # extract query
      max_qry_len = query_len.max().item()
      key_to_query = _key_states[:, 0, :max_qry_len]  # (bs, max_qry_len, emb_size_per_head)
      query_to_query = _query_states[:, 0, :max_qry_len]  # (bs, max_qry_len, emb_size_per_head)
      mask_to_query = _attention_mask[:, 0, 0, :max_qry_len]  # (bs, max_qry_len)
      input_ids_to_query = _input_ids[:, 0, :max_qry_len]  # (bs, max_qry_len)
      # retrieve
      # TODO: how many tokens to retrieve?
      # (bs, n_context * ?, seq_len, emb_size_per_head), (bs, n_context * ?, seq_len, seq_len), (bs, n_context * ?, seq_len)
      # memory_bank_topk = n_context * ?
      assert memory_bank_topk <= 2048, 'faiss does not support topk > 2048'
      __key_states, __query_states, __attention_mask, __input_ids = memory_bank.query(
        key_to_query, query_to_query, mask_to_query, input_ids_to_query,
        token_topk=2048, doc_topk=memory_bank_topk, use_random=memory_use_random)

      if memory_bank_recompute:  # run bi-encoder
        self.just_collect = True
        bi_encoder_forward(__input_ids.view(-1, seq_len), __attention_mask.view(-1, seq_len, seq_len))
        del self.just_collect
        # tensors of retrieved docs from bi-encoder
        __scores = self.retrieval['scores']
        __query_states = self.retrieval['query_states']
        __key_states = self.retrieval['key_states']
        __value_states = self.retrieval['value_states']
        __preprocessed_mask = self.retrieval['preprocessed_mask']
        # set self.retrieval back to the current docs
        self.retrieval = {
          'scores': scores,
          'query_states': query_states,
          'key_states': key_states,
          'value_states': value_states,
          'preprocessed_mask': preprocessed_mask}
        def cat_by_doc(raw_tensor, memory_bank_tensor):  # concat along the document dimension
          return torch.cat([
            raw_tensor.view(-1, n_context, *raw_tensor.size()[1:]),
            memory_bank_tensor.view(-1, memory_bank_topk, *memory_bank_tensor.size()[1:])], dim=1).view(
            -1, *raw_tensor.size()[1:])
        # (bs * (n_context + memory_bank_topk), ...)
        scores = cat_by_doc(scores, __scores)
        query_states = cat_by_doc(query_states, __query_states)
        key_states = cat_by_doc(key_states, __key_states)
        value_states = cat_by_doc(value_states, __value_states)
        preprocessed_mask = cat_by_doc(preprocessed_mask, __preprocessed_mask)
        attention_mask = cat_by_doc(attention_mask, __attention_mask.view(-1, seq_len, seq_len))
        aggregate_attention(
          self, scores, attention_mask,
          field=field, aggregation_method=aggregation_method, max_over_head=max_over_head)
      else:
        # use original query-related vectors to enable dropout
        assert memory_bank_topk % n_context == 0
        ratio = memory_bank_topk // n_context
        keep_query_rel_mask = __attention_mask[:, :, 0].unsqueeze(-1)  # (bs, n_context * ?, seq_len, 1)
        __key_states = __key_states * ~keep_query_rel_mask + _key_states.repeat(1, ratio, 1, 1) * keep_query_rel_mask
        __query_states = __query_states * ~keep_query_rel_mask + _query_states.repeat(1, ratio, 1, 1) * keep_query_rel_mask
        # combine current batch with retrieved
        # (bs * n_context * ?, seq_len, emb_size_per_head)
        __key_states = torch.cat([_key_states, __key_states], dim=1).view(-1, seq_len, emb_size)
        __query_states = torch.cat([_query_states, __query_states], dim=1).view(-1, seq_len, emb_size)
        # (bs * n_context * ?, seq_len, seq_len)
        attention_mask = torch.cat([_attention_mask, __attention_mask], dim=1).view(-1, seq_len, seq_len)
        # (bs * n_context * ?, 1 (n_heads), seq_len, seq_len)
        scores = torch.matmul(__query_states, __key_states.transpose(2, 1)).unsqueeze(1)
        scores = scores + position_bias[:, use_head_idx:use_head_idx + 1]
        aggregate_attention(
          self, scores, attention_mask,
          field=field, aggregation_method=aggregation_method, max_over_head=max_over_head, use_head_idx=use_head_idx, n_heads=n_heads)

    # add to memory bank and avoid duplicated add when checkpointing is activated
    if _key_states.requires_grad:
      # remove query
      for i in range(len(query_len)):
        pad = torch.zeros(n_context, query_len[i], emb_size).to(_key_states)  # (n_context, qry_len, emb_size_per_head)
        pad_mask = torch.zeros(n_context, query_len[i]).to(_attention_mask)  # (n_context, qry_len)
        pad_input_ids = torch.full((n_context, query_len[i]), pad_token_id).to(_input_ids)  # (n_context, qry_len)
        # always use the same seq_len
        # (n_context, seq_len, emb_size_per_head)
        key_to_add = torch.cat([_key_states[i, :, query_len[i]:], pad], dim=1)
        query_to_add = torch.cat([_query_states[i, :, query_len[i]:], pad], dim=1)
        # (n_context, seq_len)
        mask_to_add = torch.cat([_attention_mask[i, :, query_len[i], query_len[i]:], pad_mask], dim=1)
        # (n_context, seq_len)
        input_ids_to_add = torch.cat([_input_ids[i, :, query_len[i]:], pad_input_ids], dim=1)
        memory_bank.add(key_to_add, query_to_add, mask_to_add, input_ids_to_add)

    if need_retrieve:
      if memory_bank_recompute and memory_bank_additional_encode:  # used for following encoder layers
        self.preprocessed_mask = preprocessed_mask  # this will be used by following encoder layers
        return scores, query_states, key_states, value_states, preprocessed_mask
      else:
        return

  if use_hidden_states:
    scores = torch.matmul(
      hidden_states,
      hidden_states.transpose(1, 2)
    )
    scores = scores.unsqueeze(1)

  aggregate_attention(
    self, scores, attention_mask,
    field=field, aggregation_method=aggregation_method, max_over_head=max_over_head, term_weights=term_weights)
  if n_context is not None:
    ttas = self.retrieval['two_tower_attn_score']
    self.retrieval['two_tower_attn_score'] = ttas.view(-1, n_context, *ttas.size()[1:])

def cross_combine(values,  # (query_dim, doc_dim, seq_len, ...)
                  sep_lens,  # (query_dim)
                  diagonal: bool = True,
                  max_num_query: int = None,
                  max_size_per_query: int = 0):
  assert values.size(0) == sep_lens.size(0)
  query_dim, doc_dim, seq_len = values.size()[:3]
  query_values: List[torch.Tensor] = []
  doc_values: List[torch.Tensor] = []
  for value, sep_len in zip(values, sep_lens):
    query_values.append(value[:, :sep_len])
    doc_values.append(value[:, sep_len:])
  num_query = 0
  new_combs: List[List[torch.Tensor]] = []
  for qi, query_value in enumerate(query_values):
    new_combs.append([])
    for di, doc_value in enumerate(doc_values):
      new_comb = torch.cat([query_value, doc_value], dim=1)
      real_seq_len = new_comb.size(1)
      if real_seq_len < seq_len:
        size = list(new_comb.size())
        size[1] = seq_len - real_seq_len
        new_comb = torch.cat([new_comb, torch.zeros(*size).to(new_comb)], dim=1)  # padding
      elif real_seq_len > seq_len:
        new_comb = new_comb[:, :seq_len]  # truncate
      if not diagonal and qi == di:
        new_combs[-1].insert(0, new_comb)
      else:
        new_combs[-1].append(new_comb)
    if max_size_per_query:  # only keep max_size_per_query queries for each query
      assert not diagonal
      assert max_size_per_query <= len(new_combs[-1])
      new_combs[-1] = new_combs[-1][:max_size_per_query]
    num_query += 1
    if max_num_query is not None and num_query >= max_num_query:
      yield torch.stack([d for q in new_combs for d in q], dim=0)
      num_query = 0
      new_combs = []
  if num_query > 0:
    yield torch.stack([d for q in new_combs for d in q], dim=0)  # (query_dim * query_dim, doc_dim, seq_len, ...)

def cross_combine2(values,  # (query_dim, doc_dim, seq_len, seq_len, ...)
                   sep_lens,  # (query_dim)
                   diagonal: bool = True,
                   max_num_query: int = None,
                   max_size_per_query: int = 0):
  assert values.size(0) == sep_lens.size(0)
  query_dim, doc_dim, seq_len = values.size()[:3]
  query_values: List[torch.Tensor] = []
  doc_values: List[torch.Tensor] = []
  for value, sep_len in zip(values, sep_lens):
    query_values.append(value[:, :sep_len])
    doc_values.append(value[:, sep_len:, sep_len:])
  num_query = 0
  new_combs: List[List[torch.Tensor]] = []
  for qi, (query_value, sep_len) in enumerate(zip(query_values, sep_lens)):
    new_combs.append([])
    assert sep_len == query_value.size(1)
    for di, doc_value in enumerate(doc_values):
      # adjust along dim 2
      size = list(doc_value.size())
      size[2] = sep_len
      doc_value = torch.cat([torch.zeros(*size).to(doc_value), doc_value], dim=2)
      real_seq_len = doc_value.size(2)
      if real_seq_len < seq_len:
        size = list(doc_value.size())
        size[2] = seq_len - real_seq_len
        doc_value = torch.cat([doc_value, torch.zeros(*size).to(doc_value)], dim=2)  # padding
      elif real_seq_len > seq_len:
        doc_value = doc_value[:, :, :seq_len]  # truncate
      # adjust along dim 1
      new_comb = torch.cat([query_value, doc_value], dim=1)
      real_seq_len = new_comb.size(1)
      if real_seq_len < seq_len:
        ext = new_comb[:, -1:].repeat(1, seq_len - real_seq_len, *([1] * (new_comb.dim() - 2)))
        new_comb = torch.cat([new_comb, ext], dim=1)  # padding
      elif real_seq_len > seq_len:
        new_comb = new_comb[:, :seq_len]  # truncate
      if not diagonal and qi == di:
        new_combs[-1].insert(0, new_comb)
      else:
        new_combs[-1].append(new_comb)
    if max_size_per_query:  # only keep max_size_per_query queries for each query
      assert not diagonal
      assert max_size_per_query <= len(new_combs[-1])
      new_combs[-1] = new_combs[-1][:max_size_per_query]
    num_query += 1
    if max_num_query is not None and num_query >= max_num_query:
      yield torch.stack([d for q in new_combs for d in q], dim=0)
      num_query = 0
      new_combs = []
  if num_query > 0:
    yield torch.stack([d for q in new_combs for d in q], dim=0)  # (query_dim * query_dim, doc_dim, seq_len, seq_len, ...)

def in_batch_negative(self,
                      query_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
                      key_states: torch.FloatTensor,  # (bs * n_context, n_heads, seq_len, emb_size_per_head)
                      attention_mask: torch.BoolTensor,  # (bs * n_context, seq_len, seq_len)
                      n_context: int,
                      head_idx: int = None,
                      in_batch_negative_size: int = 0,
                      max_num_query_per_compute: int = None):
  # collect query, key, and attn across all gpus
  if head_idx is not None:
    global_q, global_k, global_attn = all_gather_tensors(
      query_states[:, head_idx:head_idx + 1].contiguous(), key_states[:, head_idx:head_idx + 1].contiguous(), attention_mask)
  else:
    global_q, global_k, global_attn = all_gather_tensors(query_states, key_states, attention_mask)
  # (n_q, n_context, seq_len, n_heads, emb_size_per_head)  n_q = world_size * bs
  global_q = torch.cat(global_q, dim=0).view(-1, n_context, *global_q[0].size()[1:]).transpose(2, 3)
  global_k = torch.cat(global_k, dim=0).view(-1, n_context, *global_k[0].size()[1:]).transpose(2, 3)
  # (n_q, n_context, seq_len, seq_len)
  global_attn = torch.cat(global_attn, dim=0).view(-1, n_context, *attention_mask.size()[1:])
  query_len = global_attn[:, 0, 0].sum(-1)  # (n_q)

  # cross combine
  global_q_iter = cross_combine(
    global_q, query_len, diagonal=False,
    max_num_query=max_num_query_per_compute, max_size_per_query=in_batch_negative_size)
  global_k_iter = cross_combine(
    global_k, query_len, diagonal=False,
    max_num_query=max_num_query_per_compute, max_size_per_query=in_batch_negative_size)
  global_attn_iter = cross_combine2(
    global_attn, query_len, diagonal=False,
    max_num_query=max_num_query_per_compute, max_size_per_query=in_batch_negative_size)

  for global_q in global_q_iter:
    # (n_q * <=n_q, n_context, n_heads, seq_len, emb_size_per_head)
    global_q = global_q.transpose(2, 3)
    global_k = next(global_k_iter).transpose(2, 3)
    # (n_q * <=n_q, n_context, seq_len, seq_len)
    global_attn = next(global_attn_iter)

    global_q = global_q.view(-1, *global_q.size()[2:])  # (n_q * <=n_q * n_context, n_heads, seq_len, emb_size_per_head)
    global_k = global_k.view(-1, *global_k.size()[2:])
    global_attn = global_attn.view(-1, *global_attn.size()[2:])  # (n_q * <=n_q * n_context, seq_len, seq_len)
    yield global_q, global_k, global_attn

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
        score = torch.log_softmax(score, dim=-1)
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
