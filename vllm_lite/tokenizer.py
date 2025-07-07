from transformers import AutoTokenizer
from typing import List, Any, Dict, Optional, Union

class BasicTokenizer:
    """
    A basic HuggingFace-compatible tokenizer interface for vllm_lite.
    """
    def __init__(self, model_name_or_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Any:
        return self.tokenizer(text, **kwargs)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def unk_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, 'unk_token_id', None)
    
 
    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.tokenizer.tokenize(text, **kwargs)
    
    def __len__(self) -> int:
       return len(self.tokenizer)

    @property
    def pad_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, 'pad_token_id', None)

    @property
    def eos_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, 'eos_token_id', None)

    @property
    def bos_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, 'bos_token_id', None) 