from copy import deepcopy
import os
import re
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .custom_gpt2 import GPT2ForCausalLM
from .custom_tokenizer import WarpTikTokenizer
from .client import Client, wrap_request_time, truncate_sequence
from .huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import HuggingFaceModelConfig, get_huggingface_model_config


class HuggingFaceServer:
    def __init__(self, model_config: HuggingFaceModelConfig, is_ours=False):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        if not is_ours:
            model_kwargs = {}
            if model_config.revision:
                model_kwargs["revision"] = model_config.revision
            with htrack_block(f"Loading Hugging Face model for config {model_config}"):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_id, trust_remote_code=True, **model_kwargs
                ).to(self.device)
        else:
            # model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)\
            ckpt_dir = "/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_300B_tohf/"
            _config = AutoConfig.from_pretrained(ckpt_dir)
            _config.use_flash_attn = False
            _config.scale_attn_weights = True
            self.model = GPT2ForCausalLM(_config)
            self.model.from_pretrained(ckpt_dir,
                # device_map="balanced_low_0",  # need accelerate
                offload_folder="./offload",
                load_in_8bit=False,
                torch_dtype=_config.torch_dtype, #"null"
            )
            checkpoint_file = os.path.join(ckpt_dir, "pytorch_model.bin")
            ckpt = torch.load(checkpoint_file)
            msg = self.model.load_state_dict(ckpt, strict=False)
            missing_keys = msg.missing_keys
            unexpected_keys = msg.unexpected_keys
            #### NOTICE: inv_freq, core_attention.bias, core_attention.masked_bias are buffers, which can be ignored
            if self.model._keys_to_ignore_on_load_missing is not None:
                for pat in self.model._keys_to_ignore_on_load_missing:
                    missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
            if self.model._keys_to_ignore_on_load_unexpected is not None:
                for pat in self.model._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
            print("loading msg:", "\n\tmissing:", missing_keys, "\n\tunexpected:", unexpected_keys)
            assert len(missing_keys) == 0 and len(unexpected_keys) == 0, "error in loading ckpt"
            self.model.to(self.device)
            # self.model.eval()
        if not is_ours:
            with htrack_block(f"Loading Hugging Face tokenizer model for config {model_config}"):
                self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, **model_kwargs)
        else:
            tokenizer = WarpTikTokenizer(add_bos_token=False, add_eos_token=False)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.model_max_length = 2048
            self.tokenizer = tokenizer

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt").to(self.device)
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        stop_sequences = None
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(raw_request["stop_sequences"])
            # Total number of stop words should be 1.
            assert len(stop_sequence_ids.input_ids) == 1
            # Total number of tokens in each stop word should be 1.
            assert len(stop_sequence_ids.input_ids[0]) == 1
            stop_sequences = raw_request["stop_sequences"][0]
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        if raw_request["additional"] is not None and "instance" in raw_request["additional"]:
            """
            # test for MultipleChoiceSeparateAdapter only
            """
            # Use HuggingFace's `forward` method.
            output = self.model.forward(**encoded_input)
            # output includes: dict_keys(['loss', 'logits', 'past_key_values', 'hidden_states', 'attentions', 'cross_attentions'])
            # output.logits.shape == [bs * seq_len * vocab_size] (before softmax)
            sequences = encoded_input.input_ids  # sequences = prompt = instance + reference (see as something generated) 
            encoded_input = self.tokenizer(raw_request["additional"]["instance"], return_tensors="pt").to(self.device)  # encoded_input = instance
            scores = output.logits
            scores = scores.permute(1, 0, 2).contiguous()
            raw_request["num_return_sequences"] = 1
            scores_bias = len(encoded_input.input_ids[0]) - 1 # minus one for auto-regressive 
            # A B C D E = A B (instance) C D E (reference)
            # bias = 1; scores of B -> C index; (also scores C + index D; scores D + index E; (follow lm_eval_harness, chunk the last token is deleted))
            # how this all works:
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # gpt2    \     \          \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice
        else:
            # V1:
            # Use HuggingFace's `generate` method.
            output = self.model.generate(**encoded_input, **relevant_raw_request)
            sequences = output.sequences
            scores = output.scores
            # notice it doesn't support any scores for prompt tokens, which means it returns no-sense for MultipleChoiceSeparateAdapter like hellaswag
            # https://github.com/stanford-crfm/helm/issues/1469
            scores_bias = 0
            """
            # V2:  [Debug Only; Greedy decoding]
            with torch.no_grad():
                self.model.config.use_cache = False  # to save gpu memory
                merged_scores = []
                scores_bias = 0
                assert raw_request["num_return_sequences"] == 1, "only support greedy sampling (means return seq = 1)"
                _encoded_input = deepcopy(encoded_input)
                while True:
                    assert _encoded_input.input_ids.shape[0] == 1, "only support bs=1"
                    output = self.model.forward(**_encoded_input)
                    scores = output.logits.cpu()
                    del output  # to save gpu memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    scores = scores.permute(1, 0, 2).contiguous()
                    _scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    # we do greedy sample without any temperature or top-p/k or nuclear or beam search
                    topk_logprobs = torch.topk(_scores, k=1)
                    decoded_token = self.tokenizer.convert_ids_to_tokens([topk_logprobs.indices[-1,0,0].item()])
                    _encoded_input["input_ids"] = torch.cat([_encoded_input["input_ids"], topk_logprobs.indices[-1,0,0].unsqueeze(0).unsqueeze(0).to(self.model.device)], dim=-1)
                    _encoded_input["attention_mask"] = torch.cat([_encoded_input["attention_mask"], torch.tensor([1]).unsqueeze(0).to(self.model.device)], dim=-1)

                    merged_scores.append(scores[-1,:,:])  # we only need the last one
                    if stop_sequences is not None and stop_sequences in self.tokenizer.convert_tokens_to_string(decoded_token):
                        # raw_request["stop_sequences"] == '\n'
                        # print(f" We stop due to {decoded_token}")
                        break
                    if len(merged_scores) == raw_request["max_new_tokens"]:
                        break
                sequences = _encoded_input.input_ids
                scores = merged_scores
            """
        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i + scores_bias][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                # top_logprobs_dicts.append(
                #     {
                #         self.tokenizer.convert_ids_to_tokens(k.item()): v.item()  # self.tokenizer.convert_ids_to_tokens(k.item()).decode('utf-8')
                #         for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                #     }
                # )
                # V2: force convert to utf-8 not bytes; especially for tiktoken
                _token_dict = dict()
                for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values):
                    k = self.tokenizer.convert_ids_to_tokens(k.item())
                    if type(k) == bytes:
                        try:
                            k = k.decode("utf-8")
                        except:
                            # only str(xx): "bytes:b'\\x99'" -> "bytes:\\x99"
                            k = "bytes:" + str(k).replace("b'\\", '\\').strip("'")
                    _token_dict[k] = v.item()
                top_logprobs_dicts.append(_token_dict)

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        # TODO: Get rid of the extra tokenization?
        all_tokens = [self.tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
        all_tokens = [
            [self.tokenizer.convert_tokens_to_string([token]) for token in sequence_tokens]
            for sequence_tokens in all_tokens
        ]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for (decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts) in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}

    def get_model_server_instance(self, model) -> HuggingFaceServer:
        if model not in self.model_server_instances:
            print("\n\n\n\n\n\n\n############# HERE IS OUR MODIFICATION ##############\n\n\n\n\n\n")
            model_config = get_huggingface_model_config(model)
            if model_config:
                self.model_server_instances[model] = HuggingFaceServer(model_config)
            elif model == "EleutherAI/gpt-j-6b":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("EleutherAI/gpt-j-6b")
                )
            elif model == "huggingface/gpt2":
                self.model_server_instances[model] = HuggingFaceServer(HuggingFaceModelConfig.from_string("gpt2"))
            elif model == "bigcode/santacoder":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("bigcode/santacoder")
                )
            elif model == "huggingface/starcoder":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("bigcode/starcoder")
                )
            elif model == "ours/custom_gpt2_7b":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("ours/custom_gpt2_7b"), is_ours=True
                )
            else:
                raise Exception(f"Unknown HuggingFace model: {model}")

        return self.model_server_instances[model]

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
            "additional": request.additional
        }

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        try:

            def do_it():
                return model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                    else:
                        tokens = tokenizer.tokenize(request.text)
                        # TODO(1522): Reenable this to revove "▁"
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
