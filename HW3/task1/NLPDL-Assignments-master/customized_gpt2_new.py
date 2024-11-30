from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class CustomizedGPT2Attention(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stay
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: bool = False,
        token_attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor]]]:
        # Prepare query, key, value matrices
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)  # [batch_size, seq_len, dim]
        query = self._split_heads(query, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
        
        # If past_key_value exists, concatenate with current key-value pairs
        if past_key_value is not None:
            past_key, past_value = past_key_value  # Unpack cached key and value
            key = torch.cat([past_key, key], dim=2)  # Concatenate along sequence length
            value = torch.cat([past_value, value], dim=2)
        
        # if attention_mask is not None and use_cache:
        # # For the first round of decoding, mask out the padding tokens in `key` and `value`.
        # # This ensures that these tokens are not considered during the attention computation.
        # # We will mask by applying a very large negative value.
        # # rearrange attention_mask to [batch_size, 1, seq_len, 1] for the attention computation.
        # # original: [batch_size, 1, 1, seq_len]
        #     new_attention_mask = attention_mask.permute(0, 1, 3, 2) 
        #     binary_mask = (new_attention_mask == 0).float()
        #     print(new_attention_mask)
        #     new_key = key + new_attention_mask
        #     # new_value = value + new_attention_mask
        #     new_value = value*binary_mask
        # # Update past_key_value for caching
        # else:
        #     new_key, new_value = key, value
        new_past_key_value = (key, value) if use_cache else None
        # if query.shape[2] > 1:

        #     key = new_key
        #     value = new_value

        # Self-attention mechanism
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)  # Perform attention
        # print(attn_output)
        # Merge heads and project output
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)  # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)  # Linear projection
        attn_output = self.resid_dropout(attn_output)  # Apply residual dropout

        return attn_output, new_past_key_value



class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        token_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor]]]:
        residual = hidden_states

        # Layer normalization before attention
        hidden_states = self.ln_1(hidden_states)

        # Self-attention with support for kv cache
        attn_output, new_past_key_value= self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,  # Pass in cached key-value pairs
            use_cache=use_cache,
            token_attention_mask = token_attention_mask
        )

        # Residual connection
        hidden_states = attn_output + residual

        residual = hidden_states

        # Feed-forward layer
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + feed_forward_hidden_states

        # Return updated key-value cache if use_cache is True
        if use_cache:
            return hidden_states, new_past_key_value
        else:
            return hidden_states, None 



class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = True,
        get_last_token: Optional[bool] = False,

    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds


        if get_last_token:
            # print("attention_mask:",attention_mask)
            token_attention_mask = attention_mask[:, -1:].clone()  # Only keep the last attention mask
        else:
            token_attention_mask = attention_mask.clone()
        # Prepare Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        token_attention_mask = token_attention_mask.view(batch_size, -1) if token_attention_mask is not None else None
        token_attention_mask = token_attention_mask[:, None, None, :]
        token_attention_mask = token_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        token_attention_mask = (1.0 - token_attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        # print(hidden_states[:, -1:])
        # print(hidden_states)
        if get_last_token:
            output_shape = (-1,) + (1,) + (hidden_states.size(-1),)
            # input_ids = input_ids[:, -1:]  # Only keep the last token
            # print("hidden_states_shape",hidden_states.shape)
            hidden_states = hidden_states[:, -1:]  # Only keep the last token's embeddings
        else:
            output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if past_key_values is None:
            past_key_values = [None] * len(self.h)

        new_past_key_values = [] if use_cache else None

        # Iterate over all GPT2 layer, i.e. `block`
        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache = use_cache,
                token_attention_mask = token_attention_mask
            )

            hidden_states = outputs[0]
            if use_cache:
                new_past_key_values.append(outputs[1])



        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if not use_cache:
            return {'hidden_states': hidden_states,

            }
        else:
            return {
                'hidden_states': hidden_states,
                'past_key_values': tuple(new_past_key_values),
            }



class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = True,
        get_last_token: Optional[bool] = False,
    ):
        
        # Forward pass through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            get_last_token=get_last_token,
        )

        # Extract hidden states and key-value cache from transformer output
        hidden_states = outputs["hidden_states"]
        new_past_key_values = outputs["past_key_values"] if use_cache else None

        # Prepare logits from last hidden state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'past_key_values': new_past_key_values,

        }