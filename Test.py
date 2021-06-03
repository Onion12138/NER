import json
with open('data/train.json', 'r') as f:
    with open('data/train2.json', 'w') as w:
        for line in f.readlines():
            js = json.loads(line)
            if js['text'] == '据了解，日前有媒体上刊发了题为《招商银行：投资永隆银行浮亏逾百亿港元》的文章，':
                print("found it")
            else:
                w.write(line)

# from torch.utils.data import Dataset
# import json
# from fastNLP import Vocabulary
# from transformers import BertTokenizer
# import torch
#
#
# def generate_type(type_path):
#     id2type = dict(json.load(open(type_path)))
#     type_vocab = Vocabulary(unknown=None, padding=None)
#     for id, type in id2type.items():
#         type_vocab.add_word("B-" + type)
#         type_vocab.add_word("I-" + type)
#     type_vocab.add_word("O")
#     return type_vocab
#
# data = ['{"text": "生生不息CSOL生化狂潮让你填弹狂扫", "label": {"game": {"CSOL": [[4, 7]]}}}',
#         '{"text": "此数据换算成亚洲盘罗马客场可让平半低水。", "label": {"organization": {"罗马": [[9, 10]]}}}']
#
# # type_vocab = generate_type('data/type.json')
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# string = '系列2款新作，分别为1、2代强化移植合辑的《伊苏1&2历代记(YsI&II'
# encode2 = tokenizer.encode(' '.join(list(string)), add_special_tokens=False)
# encode = [5143, 1154, 123, 3621, 3173, 868, 8024, 1146, 1166, 711, 122, 510, 123, 807, 2487, 1265, 4919, 3490, 1394, 6782, 4638, 517, 823, 5722, 122, 111, 123, 1325, 807, 6381, 113, 100, 111, 100]
# print(tokenizer.encode('[UNK]', add_special_tokens=False))
# print(len(string))
# print(len(encode))
# print(len(encode2))
# print(tokenizer.decode(encode))
# print(tokenizer.decode(encode2))
# # item = json.loads(data[0])
# # text = item['text']
# # label = item['label']
# # ids = torch.zeros(len(text))
# # for key, value in label.items():
# #     for k, v in value.items():
# #         if len(v) > 1:
# #             print(v)
# #         ids[v[0][0]+1:v[0][1]+1] = type_vocab.to_index("I-" + key)
# #         ids[v[0][0]] = type_vocab.to_index("B-" + key)
# #
# # print(ids)