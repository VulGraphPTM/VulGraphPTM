import os
import torch
import math
import dgl
import torch.nn as nn
import torch.nn.functional as f

from typing import List, Optional, Tuple, Union

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from dgl.nn import RelGraphConv
from dgl.readout import sum_nodes, softmax_nodes

from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, RobertaTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.activations import ACT2FN, gelu
from transformers.utils import logging

# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

logger = logging.get_logger(__name__)


class GlobalAttentionPooling(nn.Module):
    """
    Global Attention Pooling from `Gated Graph Sequence Neural Networks`
    """
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
    
    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            if get_attention:
                return readout, gate
            else:
                return readout


class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


## customized
class NodeEmbeddings(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        # self.cls_embeddings = nn.Embedding(1, config.hidden_size)
        # self.node_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.node_embeddings = nn.Linear(config.hidden_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.node_type_embeddings = nn.Embedding(config.type_node_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "node_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.block_size = config.block_size
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    
    def forward(
        self, inputs_embeds=None, node_type_ids=None, position_ids=None, past_key_values_length=0, mask=None
    ):
        # batch_size = inputs_embeds.size()[0]
        # node_size = inputs_embeds.size()[1]
        # cls_embeds = self.cls_embeddings(tokenizer.convert_tokens_to_ids([tokenizer.cls_token])).unsqueeze(0)
        # sep_embeds = self.cls_embeddings(tokenizer.convert_tokens_to_ids([tokenizer.sep_token])).unsqueeze(0)
        # cls_embeds = cls_embeds.repeat(batch_size, 1, 1)
        # sep_embeds = sep_embeds.repeat(batch_size, 1, 1)
        # inputs_embeds = torch.cat((cls_embeds, inputs_embeds), dim=1)
        # inputs_embeds = torch.cat((inputs_embeds, sep_embeds), dim=1)
        # padding_length = self.block_size - node_size - 2
        # pad_embeds = self.cls_embeddings(tokenizer.convert_tokens_to_ids([tokenizer.pad_token])).unsqueeze(0)

        if position_ids is None:
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, mask=mask)
        
        input_shape = inputs_embeds.size()[:-1]
        node_length = input_shape[1]

        if node_type_ids is None:
            if hasattr(self, "node_type_ids"):
                buffered_node_type_ids = self.node_type_ids[:, :node_length]
                buffered_node_type_ids_expanded = buffered_node_type_ids.expand(input_shape[0], node_length)
                node_type_ids = buffered_node_type_ids_expanded
            else:
                node_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        node_type_embeddings = self.node_type_embeddings(node_type_ids)

        # embeddings = inputs_embeds + node_type_embeddings
        # embeddings = inputs_embeds.detach().clone().requires_grad_(True)
        embeddings = self.node_embeddings(inputs_embeds)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = torch.add(embeddings, position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, mask=None):
        """
        If mask not provided. We cannot infer which are padded so just generate sequential position ids.
        Elif provided, replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored.

        Args:
            inputs_embeds: torch.Tensor (batch_size, node_length, vector_dim)

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        node_length = input_shape[1]

        if mask is not None:
            assert mask.size()[1] == node_length
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
            return incremental_indices.long() + self.padding_idx
        else:
            position_ids = torch.arange(
                self.padding_idx + 1, node_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
            )
            return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.roberta.modeling_roberta.RobertaModel with custom NodeEmbedding layer
class GraphBertModel(RobertaModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = NodeEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        node_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask: Optional[torch.Tensor] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        # only have input_embeds
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify inputs_embeds")
        
        batch_size, node_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, node_length)), device=device)
        
        if node_type_ids is None:
            if hasattr(self.embeddings, "node_type_ids"):
                buffered_node_type_ids = self.embeddings.node_type_ids[:, :node_length]
                buffered_node_type_ids_expanded = buffered_node_type_ids.expand(batch_size, node_length)
                node_type_ids = buffered_node_type_ids_expanded
            else:
                node_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # embedding_output = self.embeddings(
        #     inputs_embeds=inputs_embeds,
        #     node_type_ids=node_type_ids,
        #     position_ids=position_ids,
        #     past_key_values_length=0,
        #     mask=mask,
        # )
        embedding_output = inputs_embeds
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class GraphBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, i_dim, o_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(i_dim, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, o_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GraphBertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_nn = nn.Linear(config.hidden_size, 1)
        self.mlp = MLP(config.hidden_size, 2)
        self.act = nn.Softmax()
    
    def forward(self, outputs):
        gate = self.gate_nn(outputs)
        gate = torch.softmax(gate, dim=1)
        result = outputs * gate
        result = torch.sum(result, dim=1)
        result = self.mlp(result)
        result = self.act(result)
        return result


class GraphToBertForClassification(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.graphbert = GraphBertModel(config, add_pooling_layer=False)
        # self.classifier = GraphBertClassificationHead(config)
        self.classifier = GraphBertClassifier(config)

        self.post_init()
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        mask: Optional[torch.Tensor] = None
    ):
        outputs = self.graphbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            node_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask=mask,
        )
        attention_matrix = outputs.attentions[0]  # 注意力矩阵
        hidden_states = outputs.last_hidden_state  # 隐藏状态向量
        # hidden_states = hidden_states.unsqueeze(dim=1).repeat(1, 3, 1, 1)
        # weighted_hidden_states = torch.matmul(attention_matrix, hidden_states)
        # pooled_output = torch.sum(weighted_hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss_fct = MSELoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class RelGraphConvModel(nn.Module):
    def __init__(self,
                 input_dim, h_dim, out_dim, num_relations,
                 num_bases=-1, num_hidden_layers=1, regularizer='basis',
                 dropout=0., self_loop=False, ns_mode=False):
        super(RelGraphConvModel, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.ns_mode = ns_mode

        if self.num_bases == -1:
            self.num_bases = self.num_relations

        self.h_layers = nn.ModuleList()
        self.in_layer = RelGraphConv(self.input_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        for _ in range(self.num_hidden_layers):
            self.h_layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop))
        self.out_layer = RelGraphConv(self.h_dim, self.out_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        self.readout = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))

    def forward(self, g, cuda=False, device=None, output_ggnn=False):
        if self.ns_mode:
            # forward for neighbor sampling
            x = g[0].ndata['features']
            # input layer
            h = self.in_layer(g[0], x, g[0].edata['etype'])
            h = self.dropout(f.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(g[idx+1], h, g[idx+1].edata['etype'])
                h = self.dropout(f.relu(h))
            # output layer
            idx = len(self.h_layers) + 1
            h = self.out_layer(g[idx], h, g[idx].edata['etype'])
            return h
        else:
            x = g.ndata['features']
            e = g.edata['etype']
            if cuda:
                g = g.to(device)
                x = x.to(device)
                e = e.to(device)
            if output_ggnn:
                graph_list = dgl.unbatch(g)
                node_embeds = [graph.ndata['features'] for graph in graph_list]
            # input layer
            h = self.in_layer(g, x, e)
            h = self.dropout(f.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(g, h, e)
                h = self.dropout(f.relu(h))
            # output layer
            h = self.out_layer(g, h, e)
            if output_ggnn:
                tmp_hidden_states = h
                h = self.readout(g, h)  # shape (batch, out_dim)
                # h = self.activation(self.classifier(h)).squeeze(dim=-1)  # shape (batch,)
                g.ndata['features'] = tmp_hidden_states
                graph_list = dgl.unbatch(g)
                graph_embeds = [graph.ndata['features'] for graph in graph_list]
                return h, node_embeds, graph_embeds
            else:
                # readout function
                h = self.readout(g, h)  # shape (batch, out_dim)
                # h = self.activation(self.classifier(h)).squeeze(dim=-1)  # shape (batch,)
                return h


class MyClassificationHead(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, args.graph_embed_size)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class MyModel(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.roberta = RobertaModel.from_pretrained(args.model_path)
        self.config = config
        self.ptm_header = MyClassificationHead(config, args)
        self.gnn = RelGraphConvModel(input_dim=args.feature_size, h_dim=args.graph_embed_size, out_dim=args.graph_embed_size,
                                  num_relations=4, num_hidden_layers=1)
        self.classifier = MLP(args.graph_embed_size, 1)
        self.activation = nn.Sigmoid()

        self.post_init()
    
    def forward(self, input_ids, attention_mask, labels, g, cuda=False, device=None, output_ggnn=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        roberta_embed = self.ptm_header(sequence_output)

        gnn_embed = self.gnn(g, cuda=cuda, device=device, output_ggnn=output_ggnn)

        # all_embed = torch.cat([roberta_embed, gnn_embed], dim=-1)
        # all_embed = roberta_embed + gnn_embed
        all_embed = gnn_embed
        # logits = self.classifier(all_embed)
        # logits = torch.softmax(logits, dim=-1)
        logits = self.activation(self.classifier(all_embed)).squeeze(dim=-1)

        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss_fct = BCELoss(reduction='sum')
        loss = loss_fct(logits, labels)
        if labels is not None:
            return loss, logits
        else:
            return logits


class MyGraphModel(nn.Module):
    def __init__(self,
                 args,
                 encoder=None, tokenizer=None):
        self.args = args
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels

        self.input_dim = args.feature_size
        self.h_dim = args.graph_embed_size
        self.out_dim = args.graph_embed_size
        self.num_relations = args.num_relations
        self.num_bases = self.num_relations
        self.num_hidden_layers = args.num_hidden_layers
        self.regularizer = 'basis'
        self.dropout = nn.Dropout(args.dropout)
        self.self_loop = False

        self.h_layers = nn.ModuleList()
        self.in_layer = RelGraphConv(self.input_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        for _ in range(self.num_hidden_layers):
            self.h_layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop))
        self.out_layer = RelGraphConv(self.h_dim, self.out_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        self.readout = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))

        self.classifier = MLP(self.h_dim, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, labels, g, cuda=False, device=None, output_ggnn=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        print(last_hidden_state.size())
        assert 0 == 1
