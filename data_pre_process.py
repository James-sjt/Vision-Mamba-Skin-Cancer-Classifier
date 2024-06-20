import os
import shutil
import pandas as pd


def main():
    # classify all img
    diseases = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    path = 'Skin Cancer/Skin Cancer'
    labels = pd.read_csv('HAM10000_metadata.csv')
    list_path = os.listdir(path)

    # create folder for each disease
    for dis in diseases:
        os.mkdir(os.path.join(path, dis))

    # classify
    for i in range(len(list_path)):
        label = labels.iloc[i, 2]
        img_path = os.path.join(path, list_path[i])
        des_path = os.path.join(path, label)
        shutil.move(img_path, des_path)

    # create test set
    os.mkdir(os.path.join(path, 'test'))
    nums = {'akiec': 60, 'mel': 220, 'bkl': 220, 'nv': 1340, 'bcc': 100, 'vasc': 30, 'df': 20}
    name = 0
    for key in nums.keys():
        path_dis = path + '/' + key
        imgs = os.listdir(path_dis)
        for img in imgs[: nums[key]]:
            src_file = path_dis + '/' + img
            dest_file = 'Skin Cancer/Skin Cancer/test/' + str(name).zfill(4) + '.jpg'
            shutil.move(src_file, dest_file)
            name += 1

    print('Pre processing done!')


if __name__ == '__main__':
    main()
