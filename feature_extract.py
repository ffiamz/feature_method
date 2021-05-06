import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from face_dataset import load_PIE, load_FRDataset, preprocess
from utils import evaluate_1to1, evaluate_1toN

from sklearn.metrics import precision_score, recall_score, f1_score

def pca_fit_transform(dim, img_train):
    pca=PCA(n_components=dim)
    pca.fit(img_train)
    x_train_pca = pca.transform(img_train)
    eigen = pca.components_
    return x_train_pca, eigen

def rebuild(image_data, shape):
    # image_data: num_data * size_flatten
    fig, axes = plt.subplots(2, 8
                             , figsize=(20, 10)
                             , subplot_kw={"xticks": [], "yticks": []}  # 不要显示坐标轴
                             )
    i = 1
    for ax in axes.flat:
        if i <= 8:
            x_train_pca, V = pca_fit_transform(50*(i+1), image_data)
            ax.imshow(np.mean(image_data, axis=0).reshape(shape[0], shape[1]) +
                  np.dot(x_train_pca[0], V).reshape(shape[0], shape[1]),
                  cmap="gray")
        else:
            x_train_pca, V = pca_fit_transform(200, image_data)
            ax.imshow(V[i-8].reshape(shape[0], shape[1]), cmap="gray")
        i +=1
    plt.show()

def show_eigenface():
    images, labels = load_PIE()
    image_fltn = np.array([img.flatten() for img in images])
    print(image_fltn.shape)
    rebuild(image_fltn[100:2000], [64,64])

def svc_classify(train_x, train_y, test_x, test_y):
    clf = SVC()
    clf.fit(train_x, train_y)
    #print('score', clf.score(test_x, test_y))
    pred_y = clf.predict(test_x)
    return precision_score(test_y, pred_y, average='macro'), \
           recall_score(test_y, pred_y, average='macro'), \
           f1_score(test_y, pred_y, average='macro')


def pca_process(train_images, test_images, dim):
    train_fltnimg = np.array([img.flatten() for img in train_images])
    test_fltnimg = np.array([img.flatten() for img in test_images])
    pca = PCA(n_components=dim)
    pca.fit(train_fltnimg)
    train_x = pca.transform(train_fltnimg)
    test_x = pca.transform(test_fltnimg)
    return train_x, test_x

def pie_classify(args):
    train_images, train_labels = load_PIE()
    test_images, test_labels = load_PIE(test=True)
    train_x, test_x = pca_process(train_images, test_images, args.pca_dim)
    pre, rec, f1 = svc_classify(train_x, train_labels, test_x, test_labels)
    print(pre, rec, f1)

def face_recognization(args):
    data_path = dict(
        FR94=[f'data/Face Recognition Data/faces94/{sub_class}' for sub_class in ['female', 'male', 'malestaff']],
        FR95=['data/Face Recognition Data/faces95'],
        FR96=['data/Face Recognition Data/faces96'],
        FR_gri=['data/Face Recognition Data/grimace'],
        FR_all=None,
    )
    train_samples, test_samples = load_FRDataset(data_path[args.dataset])
    train_samples = preprocess(train_samples, align=args.align)
    test_samples = preprocess(test_samples, align=args.align)
    #train_person_num = len(train_samples)
    test_person_num = len(test_samples)
    train_x = []
    for person in train_samples:
        train_x.extend(person)
    test_x = []
    test_lens = []
    for person in test_samples:
        test_x.extend(person)
        test_lens.append(len(person))
    train_x, test_x = pca_process(train_x, test_x, args.pca_dim)
    test_array = []
    sum_len = 0
    for leng in test_lens:
        test_array.append([torch.tensor(x) for x in test_x[sum_len:sum_len+leng]])
        sum_len += leng
    best_acc, best_th = evaluate_1to1(test_array)
    print(best_acc, best_th)
    acc1, acc5 = evaluate_1toN(test_array, test_person_num)
    print(acc1, acc5)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--pca_dim', type=int, default=500)
    parser.add_argument('--dataset', choices=['PIE', 'FR94', 'FR95', 'FR96', 'FR_gri', 'FR_all'], default='PIE')
    args = parser.parse_args()
    if args.dataset == 'PIE':
        pie_classify(args)
    else:
        face_recognization(args)


