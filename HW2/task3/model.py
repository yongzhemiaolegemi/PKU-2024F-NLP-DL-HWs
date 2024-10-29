import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaModel

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Add residual connection


class RobertaForSequenceClassificationWithAdapters(RobertaForSequenceClassification):
    def __init__(self, config, adapter_size=64):
        super().__init__(config)
        self.adapter_size = adapter_size
        
        # Modify the encoder to add adapters to each layer
        for i, layer in enumerate(self.roberta.encoder.layer):
            # Add two adapters for each Transformer layer: one before and one after the feed-forward network
            layer.output.adapter_before_ffn = Adapter(config.hidden_size, adapter_size)
            layer.output.adapter_after_ffn = Adapter(config.hidden_size, adapter_size)

        # Freeze the entire model except the adapters and classification head
        self.freeze_roberta()

    def freeze_roberta(self):
        # Freeze all model parameters except adapters and classifier
        for param in self.roberta.parameters():
            param.requires_grad = False
        # Enable gradients for adapter layers and classifier
        for i, layer in enumerate(self.roberta.encoder.layer):
            for param in layer.output.adapter_before_ffn.parameters():
                param.requires_grad = True
            for param in layer.output.adapter_after_ffn.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        # Embedding 层
        embedding_output = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
        )
        
        # 获取注意力掩码
        extended_attention_mask = self.roberta.get_extended_attention_mask(attention_mask, input_ids.shape, input_ids.device)

        # 逐层应用 Transformer 层、Adapter 层和 FFN 层
        hidden_states = embedding_output
        for i, layer in enumerate(self.roberta.encoder.layer):
            # Self-attention 部分
            layer_outputs = layer.attention(
                hidden_states, 
                attention_mask=extended_attention_mask, 
                head_mask=head_mask[i] if head_mask is not None else None
            )
            attention_output = layer_outputs[0]
            
            # 在 FFN 前应用 Adapter 层
            attention_output = layer.output.adapter_before_ffn(attention_output)
            
            # FFN 部分
            intermediate_output = layer.intermediate.dense(attention_output)
            intermediate_output = layer.intermediate.intermediate_act_fn(intermediate_output)
            ffn_output = layer.output.dense(intermediate_output)
            ffn_output = layer.output.LayerNorm(ffn_output + attention_output)
            
            # 在 FFN 后应用 Adapter 层
            hidden_states = layer.output.adapter_after_ffn(ffn_output)

        # 分类头
        logits = self.classifier(hidden_states)  # 使用 [CLS] token 表示

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits, "hidden_states": hidden_states}






