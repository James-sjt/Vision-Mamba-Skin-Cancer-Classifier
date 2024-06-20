# Vision-Mamba-Skin-Cancer-Classifier
### JiatongSi, RuixiangWu, ShuhanXu
## Config Environment
Run the following code to config the environment.
```bash
$ pip install -r config.txt
```
## Download data
https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset/code
It will be 2.7G.
Unzip the data to Vision-Mamba-Skin-Cancer-Classifier-main.
Before do the next step, run the following code to check all code and data are at the right position.
```bash
$ ls
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2024/6/20     14:58                Skin Cancer
-a----         2024/6/20     16:43             91 config.txt
-a----         2024/6/20     15:04           1218 data_pre_process.py
-a----         2024/6/20     14:58         563277 HAM10000_metadata.csv
-a----         2024/6/20     15:39           2197 Loader.py
-a----         2024/6/20     12:59          11107 Mamba_v2.py
-a----         2024/6/20     12:59          15984 Mamba_v4.py
-a----         2024/6/20     13:03           3663 Model.py
-a----         2024/5/31     22:38           4503 pscan.py
-a----         2024/6/20     21:02           5193 Test.py
-a----         2024/6/20     20:33           7503 Train.py

```
## Data Pre-process
Run the following code to classify all images and devide training and testing set in a ratio of 8: 2.
```bash
$ python -u data_pre_process.py
Pre processing done!
```
## Train the model
Run the follwing code to get the help.
```bash
$ python Train.py -h
Train a Mamba-based classifier to classify skin cancer diseases.

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

```
Run the following code to train the model, the following is to train a model to classify mel and nv.
```bash
$ python -u Train.py 192 16 258 5 1 2 16 192 8 100 v4 mel nv
```
After each 10 epochs, programm will save the parameter of this model in .pth form, where you can find under Vision-Mamba-Skin-Cancer-Classifier-main. 
## Model Evaluation
Run the following code to get the help.
```bash
$ python Test.py -h
Evaluate the Mamba-based classifier to classify skin cancer diseases.

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

```
Run the folloing code to evaluate this model on testing set. (Notice: Parameters need to be the same as during training!!!)
```bash
$ python -u Test.py 192 16 258 5 1 2 16 192 8 100 v4 mel nv
Start testing...
              precision    recall  f1-score   support

         0.0       1.00      0.93      0.96       236
         1.0       0.99      1.00      0.99      1324

    accuracy                           0.99      1560
   macro avg       0.99      0.97      0.98      1560
weighted avg       0.99      0.99      0.99      1560
Test complete...

```
