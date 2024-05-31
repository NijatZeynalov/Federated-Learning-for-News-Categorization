from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def convert_labels(labels):
    label_dict = {label: idx for idx, label in enumerate(set(labels))}
    return [label_dict[label] for label in labels], label_dict

class BaseModel:
    def __init__(self):
        pass
