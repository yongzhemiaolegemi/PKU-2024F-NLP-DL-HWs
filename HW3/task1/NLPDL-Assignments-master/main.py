import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from customized_gpt2_new import CustomizedGPT2LMHeadModel

@torch.no_grad()
def customized_greedy_decoding(batch):
    # Tokenize input batch
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    input_ids = tokenized_batch['input_ids']
    attention_mask = tokenized_batch['attention_mask']
    # attention_mask = torch.ones_like(attention_mask)

    res = input_ids

    past_key_values = None  # Initialize kv cache
    start_time = time.time()
    get_last_token = False
    sequence_len = input_ids.shape[1]
    for timestep in range(sequence_len-1):
        input_ids_timestep = input_ids[:, timestep:timestep+1]
        attention_mask_timestep = attention_mask[:, :timestep+1]
        outputs = custom_model(
            input_ids = input_ids_timestep,
            attention_mask=attention_mask_timestep,
            past_key_values=past_key_values,
            use_cache=True,  # Enable caching
            get_last_token = True
        )
        logits = outputs['logits']
        past_key_values = outputs['past_key_values']  # Update kv cache
        # output_tokens = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        # res = torch.cat([res, output_tokens], dim=-1)

        
    for timestep in range(MAX_NEW_LENGTH):
        # Pass past_key_values for incremental decoding

        outputs = custom_model(
            input_ids = res,
            # input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,  # Enable caching
            get_last_token = True
        )
        
        get_last_token = True
        # Extract logits and predicted token
        logits = outputs['logits']
        past_key_values = outputs['past_key_values']  # Update kv cache
        output_tokens = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        # outputs_golden = custom_model(
        #     input_ids = res,
        #     attention_mask=attention_mask,
        #     past_key_values=None,
        #     use_cache=False,
        #     get_last_token = False
        # )
        
        # logits_golden = outputs_golden['logits']
        # output_golden_tokens = torch.argmax(logits_golden[:, -1], dim=-1, keepdim=True)
        # assert(outputs['querys'] == outputs_golden['querys'])
        # for query in outputs['querys']:
        #     print(query.shape)
        # for query in outputs_golden['querys']:
        #     print(query.shape)
        # for query, query_golden in zip(outputs['querys'], outputs_golden['querys']):
        #     print("query",query)
        #     print("query_golden",query_golden[:,-1:])
            # if get_last_token:
            #     assert(query.equal(query_golden[:,-1:]))
        # for key, key_golden in zip(outputs['keys'], outputs_golden['keys']):
        #     assert(torch.equal(key, key_golden))
        # for value, value_golden in zip(outputs['values'], outputs_golden['values']):
        #     assert(torch.equal(value, value_golden))
        # print(outputs['querys'])
        # print(outputs_golden['querys'])
        # print(outputs['keys'])
        # print(outputs_golden['keys'])
        # print(outputs['values'])
        # print(outputs_golden['values'])
        # assert torch.equal(output_tokens, output_golden_tokens), "Decoding results are not equal at time step {}".format(timestep)  # Check if the predicted tokens are equal
        # Append predicted tokens to results
        res = torch.cat([res, output_tokens], dim=-1)


        # Prepare input for the next step
        input_ids = output_tokens  # Only pass the last predicted token
        attention_mask = torch.cat([attention_mask, torch.ones_like(output_tokens)], dim=-1)

    return res, time.time() - start_time



@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        # attention_mask = torch.ones_like(tokenized_batch['attention_mask'])
        # tokenized_batch['attention_mask'] = attention_mask
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 16
    times = [0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)

        times[0] += golden_wo_cache_time
        times[1] += custom_time
        decoded_golden_wo_cache = tokenizer.batch_decode(golden_wo_cache_res, skip_special_tokens=True)
        decoded_custom = tokenizer.batch_decode(custom_res, skip_special_tokens=True)
        for i in range(len(decoded_golden_wo_cache)):
            print("golden:",decoded_golden_wo_cache[i])
            print("custom:",decoded_custom[i])
        assert torch.equal(golden_wo_cache_res, custom_res), "Decoding results are not equal"

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])
