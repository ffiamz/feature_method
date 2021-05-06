import os
import argparse
import logging
import torch
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(model, loader):
    def pred(x):
        _, id = torch.max(softmax(x, dim=-1), dim=1)
        return id.cpu().numpy()

    true_ys = np.array([], dtype=np.int)
    pred_ys = np.array([], dtype=np.int)
    for x, y in loader:
        x = x.to(model.device())
        y = y.to(model.device())

        _, y_hat = model(x)
        true_ys = np.concatenate((true_ys, y.squeeze(-1).cpu().numpy()))
        pred_ys = np.concatenate((pred_ys, pred(y_hat)))

    return precision_score(true_ys, pred_ys, average='macro'), \
           recall_score(true_ys, pred_ys, average='macro'), \
           f1_score(true_ys, pred_ys, average='macro')


def calc_feat(model, loader, tot_class):
    feat_arr = [[] for _ in range(tot_class)]
    for imgs, labels in loader:
        imgs = imgs.to(model.device())
        labels = labels.to(model.device())
        # print(imgs.shape)
        with torch.no_grad():
            feats, _ = model(imgs)
        # print(feats.shape)

        for i in range(len(feats)):
            feat_arr[labels[i].squeeze(-1).item()].append(feats[i])

    return feat_arr


def iter_feat(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            yield i, arr[i][j]


def rand_one_feat(rand, arr):
    i = rand.randint(len(arr))
    j = rand.randint(len(arr[i]))
    return i, arr[i][j]


def rand_one_feat_by_person(rand, arr, i):
    j = rand.randint(len(arr[i]))
    return i, arr[i][j]


def calc_dis(x, y):
    return (x / x.norm() - y / y.norm()).norm().item()


def evaluate_1to1(feat_arr):
    rand = np.random.RandomState(0)
    pos_n, neg_n, dis_arr, labels = 0, 0, [], []
    while pos_n < 3000 or neg_n < 3000:
        person1, feat1 = rand_one_feat(rand, feat_arr)
        person2, feat2 = rand_one_feat(rand, feat_arr)
        while pos_n == 3000 and person1 == person2:
            person2, feat2 = rand_one_feat(rand, feat_arr)
        while neg_n == 3000 and person1 != person2:
            person2, feat2 = rand_one_feat_by_person(rand, feat_arr, person1)
        if person1 == person2:
            pos_n += 1
            dis_arr.append(calc_dis(feat1, feat2))
            labels.append(1)
        else:
            neg_n += 1
            dis_arr.append(calc_dis(feat1, feat2))
            labels.append(0)

    dis_arr, labels = np.asarray(dis_arr), np.asarray(labels)
    best_acc, best_th = 0, 0
    for th in dis_arr:
        preds = (dis_arr < th)
        acc = np.mean((preds == labels).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_acc, best_th


def evaluate_1toN(feat_arr, tot_class):
    rand = np.random.RandomState(0)
    tot, acc_top1, acc_top5 = 0, 0, 0
    for person0, feat0 in iter_feat(feat_arr):
        tot += 1
        dis_arr = []
        for person_i in range(tot_class):
            _, feat_i = rand_one_feat_by_person(rand, feat_arr, person_i)
            dis_arr.append(-calc_dis(feat0, feat_i))

        dis_arr = torch.FloatTensor(dis_arr).to(feat0.device)
        if person0 in torch.argmax(dis_arr):
            acc_top1 += 1
        if dis_arr.shape[0] >= 5 and (person0 in torch.topk(dis_arr, 5)[1]):
            acc_top5 += 1

    return acc_top1 / tot, acc_top5 / tot
