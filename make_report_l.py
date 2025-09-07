from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

model = AutoModelForCausalLM.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", device_map="auto", torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0")

# pad_token が未設定の場合は eos を使用
if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

streamer = TextStreamer(tokenizer)

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

# attention_mask を付与
attention_mask = torch.ones_like(input_ids)

output_ids = model.generate(
    input_ids,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    streamer=streamer
) 