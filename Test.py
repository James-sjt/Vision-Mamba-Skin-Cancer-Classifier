"""Evaluate the Mamba-based classifier to classify skin cancer diseases.

Usage:
  Test.py <d_model> <state_size> <seq_len> <batch_size> <is_dropout>
  <num_cls> <patch_size> <emb_out_features> <num_blocks> <epochs> <version> <cls_1> <cls_2>
  Test.py (-h | --help)

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

from Model import Model
import torch
from Loader import loader_test
from docopt import docopt
from sklearn.metrics import classification_report
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main(config):
    model = Model(config.d_model, config.state_size, config.seq_len,
                  config.batch_size, config.is_dropout,
                  config.patch_size, config.emb_out_features, config.num_cls,
                  config.num_blocks, config.version).to(device)

    path = 'Epoch_' + str(config.epochs + 1) + '_' + config.cls_1 + '&' + config.cls_2 + '.pth'
    model.load_state_dict(torch.load(path, map_location=device))

    print('Start testing...')
    nums = {'akiec': 60, 'mel': 220, 'bkl': 220, 'nv': 1340, 'bcc': 100, 'vasc': 30, 'df': 20}
    flags = {'akiec': 0, 'mel': 60, 'bkl': 280, 'nv': 500, 'bcc': 1840, 'vasc': 1940, 'df': 1970}
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
        images, label = loader_test(config.batch_size, flag, config.cls_1, config.cls_2)
        op = model(images)
        for j in range(config.batch_size):
            pre[i * config.batch_size + j] = torch.argmax(op[j])
            labels[i * config.batch_size + j] = torch.argmax(label[j])
        flag += config.batch_size

    print(classification_report(pre, labels))
    print('Test complete...')
    torch.cuda.empty_cache()


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
