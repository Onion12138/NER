class Config:
    def __init__(self, config):
        self.dropout = config['dropout']
        self.bert_dim = config['bert_dim']
        self.hidden = config['hidden']
        self.layers = config['layers']
        self.num_classes = config['num_classes']