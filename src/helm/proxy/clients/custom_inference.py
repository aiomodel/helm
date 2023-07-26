            
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from custom_gpt2_debug import GPT2ForCausalLM
from custom_tokenizer import WarpTikTokenizer

ckpt_dir = "/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_300Bof1000B_tohf/"
_config = AutoConfig.from_pretrained(ckpt_dir)
_config.use_flash_attn = False
_config.scale_attn_weights = True
model = GPT2ForCausalLM(_config)
model.from_pretrained(ckpt_dir,
    # device_map="balanced_low_0",  # need accelerate
    offload_folder="./offload",
    load_in_8bit=False,
    torch_dtype=_config.torch_dtype, #"null"
)
checkpoint_file = os.path.join(ckpt_dir, "pytorch_model.bin")
ckpt = torch.load(checkpoint_file)
msg = model.load_state_dict(ckpt, strict=False)
missing_keys = msg.missing_keys
unexpected_keys = msg.unexpected_keys
#### NOTICE: inv_freq, core_attention.bias, core_attention.masked_bias are buffers, which can be ignored
if model._keys_to_ignore_on_load_missing is not None:
    for pat in model._keys_to_ignore_on_load_missing:
        missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
if model._keys_to_ignore_on_load_unexpected is not None:
    for pat in model._keys_to_ignore_on_load_unexpected:
        unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
print("loading msg:", "\n\tmissing:", missing_keys, "\n\tunexpected:", unexpected_keys)
assert len(missing_keys) == 0 and len(unexpected_keys) == 0, "error in loading ckpt"
model.to("cuda:0")
# self.model.eval()

tokenizer = WarpTikTokenizer(add_bos_token=False, add_eos_token=False)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048
tokenizer = tokenizer

# ' @-@ unsubstituted <unk> is quite versatile in the breadth of possible nucleophiles and corresponding products . <unk> may be derived from the'
encoded_input = {
    "input_ids": torch.tensor([[  571,    12,    31, 80304,  3781,  2844,   366,  3200,    29,   374,
          5115, 33045,   304,   279, 58321,   315,  3284, 31484,  5237,  3742,
           323, 12435,  3956,   662,   366,  3200,    29,  1253,   387, 14592,
           505,   279]], device=model.device)
}

encoded_input["attention_mask"] = torch.ones_like(encoded_input["input_ids"])
output = model.forward(**encoded_input)
logits = output.logits[:,:-1,:]
label = encoded_input["input_ids"][:,1:]
loss = torch.nn.functional.cross_entropy(logits.squeeze(), label.squeeze())
print(loss)