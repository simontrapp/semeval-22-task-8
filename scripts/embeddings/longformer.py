import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer

config = LongformerConfig.from_pretrained('longformer-base-4096/')
config.attention_mode = 'sliding_chunks'   # 'n2': for regular n2 attention, 'tvm': a custom CUDA kernel implementation of our sliding window attention, 'sliding_chunks': a PyTorch implementation of our sliding window attention
DEFAULT_MODEL = Longformer.from_pretrained('longformer-base-4096/', config=config)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.model_max_length = DEFAULT_MODEL.config.max_position_embeddings


def encode_longformer(text: str, model=DEFAULT_MODEL):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # batch of size 1
    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to local attention
    # attention_mask[:, [1, 4, 21, ]] = 2  # Set global attention based on the task. For example, classification: the <s> token, QA: question tokens
    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)
    output = model(input_ids, attention_mask=attention_mask)[0]
    return output
