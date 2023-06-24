import torch
from copy import deepcopy
import os
import re
from custom_gpt2 import GPT2ForCausalLM
from custom_tokenizer import WarpTikTokenizer            
from transformers import AutoConfig

device: str = "cuda:0"
tokenizer = WarpTikTokenizer(add_bos_token=False, add_eos_token=False)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

ckpt_dir = "/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_300B_tohf/"
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
model.to(device)


#######
#  DEBUG .generate()
#######
# encoded_input = tokenizer(["What's the capital of the Unit States?"], return_tensors="pt").to(device)
# encoded_input = tokenizer(["Passage: Interleague play in Major League Baseball refers to regular-season baseball games played between an American League (AL) team and a National League (NL) team. Interleague play was first introduced in the 1997 Major League Baseball season. Prior to that, matchups between AL teams and NL teams occurred only during spring training, the All-Star Game, other exhibition games (such as the Hall of Fame Game in Cooperstown, New York), and the World Series. Unlike modern interleague play, none of these contests, except for the World Series, counted toward official team or league records.\nQuestion: Do the al and nl play each other?\nAnswer: Yes\n\nPassage: Elmendorf Air Force Base (IATA: EDF, ICAO: PAED, FAA LID: EDF) is a United States military facility in Anchorage, the largest city in Alaska. Originally known as Elmendorf Field, it became Elmendorf Air Force Base after World War II, and in 2010 it merged with nearby Fort Richardson to form Joint Base Elmendorf-Richardson.\nQuestion: Is there an air force base in anchorage alaska?\nAnswer:"], return_tensors="pt").to(device)
# encoded_input = tokenizer(["Passage: Interleague play in Major League Baseball refers to regular-season baseball games played between an American League (AL) team and a National League (NL) team. Interleague play was first introduced in the 1997 Major League Baseball season. Prior to that, matchups between AL teams and NL teams occurred only during spring training, the All-Star Game, other exhibition games (such as the Hall of Fame Game in Cooperstown, New York), and the World Series. Unlike modern interleague play, none of these contests, except for the World Series, counted toward official team or league records. Question: Do the al and nl play each other? Answer: Yes. Passage: Elmendorf Air Force Base (IATA: EDF, ICAO: PAED, FAA LID: EDF) is a United States military facility in Anchorage, the largest city in Alaska. Originally known as Elmendorf Field, it became Elmendorf Air Force Base after World War II, and in 2010 it merged with nearby Fort Richardson to form Joint Base Elmendorf-Richardson. Question: Is there an air force base in anchorage alaska? Answer:"], return_tensors="pt").to(device)
encoded_input = tokenizer(["Passage: Elmendorf Air Force Base (IATA: EDF, ICAO: PAED, FAA LID: EDF) is a United States military facility in Anchorage, the largest city in Alaska. Originally known as Elmendorf Field, it became Elmendorf Air Force Base after World War II, and in 2010 it merged with nearby Fort Richardson to form Joint Base Elmendorf-Richardson.\nQuestion: Is there an air force base in anchorage alaska?\nAnswer:"], return_tensors="pt").to(device)

def compare_oneshot_onebyone(encoded_input, k=5):
    relevant_raw_request = {'temperature': 0.0001,
    'num_return_sequences': 1,
    'max_new_tokens': 2,
    'top_p': 1,
    'additional': None,
    'do_sample': True,
    'use_cache': True,
    'return_dict_in_generate': True,
    'output_scores': True,
    'eos_token_id': 198}

    # relevant_raw_request["temperature"] = 0.0
    # relevant_raw_request["do_sample"] = False
    relevant_raw_request["max_new_tokens"] = 1
    _encoded_input = deepcopy(encoded_input)

    ## One-by-One
    print("\nBEGIN ONE BY ONE\n")
    for i in range(k):
        output = model.generate(**_encoded_input, **relevant_raw_request)
        _encoded_input["input_ids"] = output.sequences
        _encoded_input["attention_mask"] = torch.cat([_encoded_input["attention_mask"], torch.tensor([1]).unsqueeze(0).to(model.device)], dim=-1)
        print(tokenizer.convert_ids_to_tokens(_encoded_input["input_ids"][0][-10-k:].tolist()))

    ## One-shot
    print("\nBEGIN ONE SHOT\n")
    relevant_raw_request["max_new_tokens"] = k
    _encoded_input = deepcopy(encoded_input)
    output = model.generate(**_encoded_input, **relevant_raw_request)
    print(tokenizer.convert_ids_to_tokens(output.sequences[0][-10-k:].tolist()))


compare_oneshot_onebyone(encoded_input, k=10)

import IPython
IPython.embed()

## Case 1 is normal; But Case 2 and 3 are abnormal..?; Also Case 4 is abnormal. [The first two is same, but different at the third]
## about past_key_values: (layer1, layer..); layer1 = (key, value); key = bs * head num * tokens * head dim
# In [1]: model_inputs['input_ids']
# Out[1]: tensor([[11]], device='cuda:0')

# In [2]: model_inputs['position_ids']
# Out[2]: tensor([[107]], device='cuda:0')

# In [9]: type(model_inputs['past_key_values'][0][0])
# Out[9]: torch.Tensor

# In [10]: model_inputs['past_key_values'][0][0].shape
# Out[10]: torch.Size([1, 32, 107, 128])