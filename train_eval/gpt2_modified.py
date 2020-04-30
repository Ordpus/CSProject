import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.modeling_gpt2 import Block


class GPT2REModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2REModel, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.ent = nn.Linear(config.n_embd, config.n_embd)
        self.pos = nn.Linear(config.n_embd, config.n_embd)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids, attention_mask, position_ids, token_type_ids, past=None):
        device = input_ids.device

        inputs_embeds = torch.zeros(*input_ids.shape, self.wte.embedding_dim, dtype=self.wte.weight.dtype,
                                    device=device)
        position_embeds = torch.zeros(*inputs_embeds.shape, dtype=self.wpe.weight.dtype, device=device)
        token_type_embeds = self.wte(token_type_ids)

        for i in range(input_ids.shape[0]):
            inp = input_ids[i]
            pos = position_ids[i]
            type_id = token_type_ids[i]
            e1, e2 = inp[type_id == 0], inp[type_id == 1]
            ep1, ep2 = pos[type_id == 0], pos[type_id == 1]
            e1e, e2e = self.wte(e1), self.wte(e2)
            e1p, e2p = self.wpe(ep1), self.wpe(ep2)
            embd = self.ent(torch.cat((e1e, e2e)))
            pos_embd = self.pos(torch.cat((e1p, e2p)))
            if 2 in type_id:
                embd = torch.cat((embd, self.wte(inp[type_id == 2])))
                pos_embd = torch.cat((pos_embd, self.wpe(pos[type_id == 2])))
            inputs_embeds[i] = embd
            position_embeds[i] = pos_embd

        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (attention_mask - 1.0) * 10000.0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        input_shape = inputs_embeds.size()[:-1]

        output_shape = input_shape + (hidden_states.size(-1),)

        if past is None:
            past = [None] * len(self.h)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask)

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class GPT2LMREModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2LMREModel, self).__init__(config)
        self.transformer = GPT2REModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None, labels=None, past=None):
        from global_constants import ignore_index
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                               position_ids=position_ids, token_type_ids=token_type_ids, past=past)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
