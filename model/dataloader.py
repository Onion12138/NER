from torch.utils.data import Dataset
import json
from fastNLP import Vocabulary
from fastNLP import TorchLoaderIter
from fastNLP import RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import torch


def generate_type(type_path):
    id2type = dict(json.load(open(type_path)))
    type_vocab = Vocabulary(unknown=None, padding=None)
    type_vocab.add_word("O")
    for id, type in id2type.items():
        type_vocab.add_word("B-" + type)
        type_vocab.add_word("I-" + type)
    return type_vocab



class MyDataset(Dataset):
    def __init__(self, data_path, type_path):
        super(MyDataset, self).__init__()
        self.data = []
        with open(data_path, 'r') as f:
            for json_line in f.readlines():
                self.data.append(json.loads(json_line))
        self.type_vocab = generate_type(type_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # self.vocabulary = self.tokenizer.get_vocab()

    def __getitem__(self, idx):
        item = dict(self.data[idx])
        text = item['text']
        char_list = list(text)
        # for c in char_list:
        #     if c not in self.tokenizer.get_vocab():
        #         self.tokenizer.add_tokens()
        #         self.tokenizer.add_special_tokens({"additional_special_tokens": c})
        #         print(self.tokenizer.all_special_tokens)
                # self.tokenizer.add_tokens(c)
                # self.vocabulary[c] = self.tokenizer.get_vocab()[c]

        tokenized = self.tokenizer(' '.join(char_list), add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        label = item['label']
        seq_label = torch.zeros(len(text))
        entity = set()
        for key, value in label.items():
            for k, v_list in value.items():
                entity.add(k)
                for v in v_list:
                    seq_label[v[0]] = self.type_vocab.to_index("B-" + key)
                    if v[0] == v[1]:
                        continue
                    seq_label[v[0] + 1:v[1] + 1] = self.type_vocab.to_index("I-" + key)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        seq_label = torch.tensor(attention_mask, dtype=torch.long)
        return input_ids, attention_mask, seq_label, text, entity

    def __len__(self):
        return len(self.data)


def my_collate_fn(batch):
    input_ids, attention_mask, seq_label, text, entity = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    seq_label = pad_sequence(seq_label, batch_first=True)
    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_label": seq_label
        }, {
        "seq_label": seq_label,
        "text": text,
        "entity": entity
    }


dataset = MyDataset('../data/train2.json', '../data/type.json')
dataset_iter = TorchLoaderIter(dataset=dataset, collate_fn=my_collate_fn, batch_size=8, sampler=RandomSampler())
# dataset = MyDataset('../data/train2.json', '../data/type.json')


#
# dataset.set_input('input_ids')
#
# for data in dataset:
#     pass
