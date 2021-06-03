import torch.nn as nn
from fastNLP.modules.decoder import ConditionalRandomField
from transformers import BertModel
from fastNLP.core.const import Const
import torch.nn.functional as F


class NERModel(nn.Module):
    def __init__(self, config):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(config.bert_dim, hidden_size=config.hidden, num_layers=config.layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden * 2, config.num_classes)
        self.crf = ConditionalRandomField(config.num_classes, include_start_end_trans=True, allowed_transitions=None)

    def forward(self, input_ids, attention_mask, seq_label):

        x = self.bert(input_ids, attention_mask=attention_mask)[0]

        x, _ = self.lstm(x)

        x = self.fc(x)

        x = self.dropout(x)

        logits = F.log_softmax(x, dim=-1)

        loss = self.crf(logits, seq_label, attention_mask).mean()

        return {Const.LOSS: loss}

    def predict(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)[0]

        x, _ = self.lstm(x)

        x = self.fc(x)

        x = self.dropout(x)

        logits = F.log_softmax(x, dim=-1)

        return self.crf.viterbi_decode(logits, attention_mask)

