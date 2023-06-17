from abc import ABC
from abc import abstractmethod
import tiktoken

################## ORIGINAL TIKTOKEN ######################
EXTRA_TOKEN_NUM = 0

class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))

class TikTokenizer(AbstractTokenizer):
    def __init__(self, tokenizer_name_or_path, vocab_extra_ids=0, update_special=True):
        name = tokenizer_name_or_path
        super().__init__(name)
        self.tokenizer = tiktoken.get_encoding(tokenizer_name_or_path)
        self.update_special = update_special
        eod_token = '<|endoftext|>'
        if update_special:
            src_spe = self.tokenizer._special_tokens
            update_spe = {}
            for k, v in src_spe.items():
                update_spe[k.replace("<|", "<|msra_")] = v
            self.tokenizer = tiktoken.Encoding(name=name+"_msra", 
                                            pat_str=self.tokenizer._pat_str, 
                                            mergeable_ranks=self.tokenizer._mergeable_ranks, 
                                            special_tokens=update_spe)
            self.tokenizer.encode("just a sanity check <|endoftext|>")
            eod_token = '<|msra_endoftext|>'
        assert vocab_extra_ids==0, "For Now, tiktoken do not support modifying vocabulary during inference!"
        self.v_2_i = self.tokenizer._mergeable_ranks
        self.i_2_v = {v: k for k, v in self.v_2_i.items()}
        assert len(self.v_2_i) == len(self.i_2_v), "Strange bug"
        self.eod_id = self.tokenizer._special_tokens[eod_token]

    @property
    def vocab_size(self):
        return self.tokenizer.max_token_value

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        """We may not need it"""
        return self.v_2_i

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        """We may not need it"""
        return self.i_2_v

    def tokenize(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def detokenize_safer(self, token_ids):
        return self.tokenizer.decode_bytes(token_ids)

    @property
    def eod_token(self):
        return "<|msra_endoftext|>" if self.update_special else "<|endoftext|>"

    @property
    def pad(self):
        return self.eod_id
                                      
    @property
    def eod(self):
        return self.eod_id

original_tokenizer = TikTokenizer("cl100k_base", vocab_extra_ids=0, update_special=True)
################################################################

################# New Tiktoken ##################
"""Tokenization classes for TikToken."""
import os
from typing import Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "nothing_to_save"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-tokenizer": 2048,
}

class WarpTikTokenizer(PreTrainedTokenizer):
    """
    Construct a tiktoken tokenizer. [Internal Use it]
    # write it based on llama tokenizer

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = None # PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = None # PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        unk_token=None,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        assert unk_token is None and bos_token is None and eos_token is None and pad_token is None
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.tokenizer = TikTokenizer("cl100k_base", vocab_extra_ids=0, update_special=True)
        assert len(self.all_special_tokens) == 0

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, d):
        raise NotImplementedError

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.tokenizer.vocab_size

    @property
    def eos_token(self) -> str:
        return self.tokenizer.eod_token

    @property
    def bos_token(self) -> str:
        return self.tokenizer.eod_token
    
    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.eod

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eod
    
    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.eod
    
    def get_vocab(self):
        """Returns vocab as a dict"""
        return self.tokenizer.vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        # original sp.encode('This is a test', out_type=str)
        return [self._convert_id_to_token(_token) for _token in self.tokenizer.tokenize(text)]

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token == self.eos_token:
            return self.eos_token_id
        return self.tokenizer.vocab[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index == self.eos_token_id:
            return self.eos_token
        return self.tokenizer.inv_vocab[index]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        if len(tokens) == 1:
            # just return its binary format
            try:
                tokens = tokens[0].decode("utf-8")
            except:
                # only str(xx): "bytes:b'\\x99'" -> "bytes:\\x99"
                tokens = "bytes:" + str(tokens[0]).replace("b'\\", '\\').strip("'")
            return tokens
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += b"".join(current_sub_tokens).decode("utf-8") + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += b"".join(current_sub_tokens).decode("utf-8")
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # we don't need saving tiktoken
        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output