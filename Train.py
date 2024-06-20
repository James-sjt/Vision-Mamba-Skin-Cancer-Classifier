"""Train a Mamba-based classifier to classify skin cancer diseases.

Usage:
  Train.py <d_model> <state_size> <seq_len> <batch_size> <is_dropout>
  <num_cls> <patch_size> <emb_out_features> <num_blocks> <epochs> <version> <cls_1> <cls_2>
  Train.py (-h | --help)

General options:
  -h --help             Show this screen.

Arguments:
  <d_model>             The dimension of Mamba model.
  <state_size>          The parameter 'n' in Mamba paper.
  <seq_len>             The sequence length of Mamba model.
  <batch_size>          The batch size for training.
  <is_dropout>          Whether to use dropout or not in training. e.g. 0: False, 1: True.
  <num_cls>             The number of class the model is to classify.
  <patch_size>          The patch size for embedding.
  <emb_out_features>    The dimension for the output of embedding.
  <num_blocks>          The number of vision mamba block in model.
  <epochs>              Number of epochs.
  <version>             The version of model. e.g. v4
  <cls_1>               The first class from ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'].
  <cls_2>               The second class from ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'].

"""


import torch.nn as nn
from Model import Model
import torch.utils.data
import torch.nn.init as init
from Loader import loader_train, loader_test
from sklearn.metrics import classification_report
import time
from tqdm import tqdm
from docopt import docopt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


class model_config:
    def __init__(self, d_model, state_size, seq_len, batch_size, is_dropout, num_cls, patch_size, emb_out_features
                 , num_blocks, epochs, version, cls_1, cls_2):
        self.d_model = int(d_model)
        self.state_size = int(state_size)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.is_dropout = bool(int(is_dropout))  # Assuming is_dropout is passed as 0 or 1
        self.num_cls = int(num_cls)
        self.patch_size = int(patch_size)
        self.emb_out_features = int(emb_out_features)
        self.num_blocks = int(num_blocks)
        self.epochs = int(epochs)
        self.version = version
        self.cls_1 = cls_1
        self.cls_2 = cls_2

    def __str__(self):
        return (f"TrainingConfig(d_model={self.d_model}, state_size={self.state_size}, seq_len={self.seq_len}, "
                f"batch_size={self.batch_size}, is_dropout={self.is_dropout}, num_cls={self.num_cls}, "
                f"patch_size={self.patch_size}, emb_out_features={self.emb_out_features},"
                f" num_blocks={self.num_blocks}, "f"epochs={self.epochs}, version={self.version},"
                f" cls_1={self.cls_1}, cls_2={self.cls_2})")


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


def main(config):
    model = Model(config.d_model, config.state_size, config.seq_len,
                  config.batch_size, config.is_dropout,
                  config.patch_size, config.emb_out_features, config.num_cls,
                  config.num_blocks, config.version).to(device)

    print('start training...')
    print('Total parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.apply(init_weights)

    # loss_function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    idx_helper = [0, 0]
    for epoch in range(config.epochs):
        train_pres, train_labels = torch.zeros(1000), torch.zeros(1000)
        for i in tqdm(range(int(1000 / config.batch_size))):
            images, labels, idx_helper = loader_train(config.batch_size, idx_helper, config.cls_1, config.cls_2)
            outputs = model(images)
            loss = criterion(outputs, labels).to(device)
            for j in range(config.batch_size):
                if outputs[j, 1] > 0.5:
                    train_pres[i * config.batch_size + j] = 1
                if labels[j, 1] == 1:
                    train_labels[i * config.batch_size + j] = 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time.sleep(0.5)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}/{config.epochs}, Learning Rate: {current_lr}')
        print('*' * 55)
        print('Summary of ' + str(epoch + 1) + ' train set: ')
        print(classification_report(train_pres, train_labels))
        print('*' * 55)
        print('')

        if (epoch + 1) % 10 == 0:
            nums = {'akiec': 60, 'mel': 220, 'bkl': 220, 'nv': 1340, 'bcc': 100, 'vasc': 30, 'df': 20}
            flags = {'akiec': 0, 'mel': 60, 'bkl': 280, 'nv': 500, 'bcc': 1840, 'vasc': 1940, 'df': 1970}
            for param in model.parameters():
                param.requires_grad = False
            print('************ Validation set Epoch: ', epoch + 1, ' *****************')
            total_num = nums[config.cls_1] + nums[config.cls_2]
            pre, labels = torch.zeros(total_num), torch.zeros(total_num)
            flag = flags[config.cls_1]
            for i in range(int(nums[config.cls_1] / config.batch_size)):
                images, label = loader_test(config.batch_size, flag, config.cls_1, config.cls_2)
                op = model(images)
                for j in range(config.batch_size):
                    pre[i * config.batch_size + j] = torch.argmax(op[j])
                    labels[i * config.batch_size + j] = torch.argmax(label[j])
                flag += config.batch_size

            flag = flags[config.cls_2]
            for i in range(int(nums[config.cls_2] / config.batch_size)):
                images, label = loader_test(config.batch_size, flag, config.clas_1, config.cls_2)
                op = model(images)
                for j in range(config.batch_size):
                    pre[i * config.batch_size + j] = torch.argmax(op[j])
                    labels[i * config.batch_size + j] = torch.argmax(label[j])
                flag += config.batch_size

            print(classification_report(pre, labels))
            torch.save(model.state_dict(), 'Epoch_' + str(epoch + 1) + '.pth')
            for param in model.parameters():
                param.requires_grad = True
            print('*' * 55)
            print('')
    print('training finished...')


if __name__ == '__main__':
    arguments = docopt(__doc__)

    config = model_config(
        d_model=arguments['<d_model>'],
        state_size=arguments['<state_size>'],
        seq_len=arguments['<seq_len>'],
        batch_size=arguments['<batch_size>'],
        is_dropout=arguments['<is_dropout>'],
        num_cls=arguments['<num_cls>'],
        patch_size=arguments['<patch_size>'],
        emb_out_features=arguments['<emb_out_features>'],
        num_blocks=arguments['<num_blocks>'],
        epochs=arguments['<epochs>'],
        version=arguments['<version>'],
        cls_1=arguments['<cls_1>'],
        cls_2=arguments['<cls_2>']
    )

    main(config)

