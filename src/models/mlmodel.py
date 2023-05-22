from pathlib import Path
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

class MLModel:
    def __init__(self):
        model_path = Path("../..", "ML", "deeplearning", "pretrained", "models", "spam_no_spam_model").resolve()
        tokenizer_path = Path("../..", "ML", "deeplearning", "pretrained", "tokenizers", "spam_no_spam_tokenizer").resolve()
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)