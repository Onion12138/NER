from fastNLP import Trainer, LossInForward
from model.ner import NERModel
import torch
from transformers import AdamW
from model.dataloader import dataset_iter, generate_type
from model.config import Config


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_dict = {'dropout': 0.5, 'bert_dim': 768, 'hidden': 100, 'layers': 2, 'num_classes': 21}
    config = Config(config_dict)
    model = NERModel(config)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    train_data = dataset_iter

    trainer = Trainer(model=model, loss=LossInForward(), optimizer=optimizer, train_data=train_data)
    trainer.train()

    type_vocab = generate_type('../data/type.json')


    for batch_x, batch_y in dataset_iter:
        prediction = model.predict(*batch_x)
        text = batch_y['text']
        entity = batch_y['entity']
        predicted_entity = set()
        all_entity = set()

        for i in range(len(prediction)):
            for j in range(0, len(prediction[i])):
                if prediction[i][j] != 0 and prediction[i][j] % 2 == 1:
                    prediction_id = prediction[i][j]
                    prediction_position = j
                    while j + 1 < len(prediction[i]) and prediction[i][j+1] == prediction_id + 1:
                        j += 1
                    predicted_entity.add(text[prediction_position: j+1])

        true_positive = predicted_entity & entity

        pre = len(true_positive) / len(predicted_entity)

        recall = len(true_positive) / len(entity)

        print(pre, recall)




