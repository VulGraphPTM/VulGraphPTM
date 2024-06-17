import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():
    with open('./prediction.txt', 'r') as fp:
        preds = fp.readlines()
    l1_p0 = []
    l0_p1 = []
    l0_p0 = []
    l1_p1 = []
    labels = []
    logits = []
    for pred in preds:
        idx, label, logit, probe, frm, value = pred.split('\t')
        label = int(label.split(':')[-1].strip())
        logit = int(logit.split(':')[-1].strip())
        labels.append(label)
        logits.append(logit)
        if label == 1:
            if logit == 1:
                l1_p1.append(idx)
            elif logit == 0:
                l1_p0.append(idx)
        elif label == 0:
            if logit == 1:
                l0_p1.append(idx)
            elif logit == 0:
                l0_p0.append(idx)
    correct = l0_p0 + l1_p1
    acc = accuracy_score(labels, logits)
    f1 = f1_score(labels, logits)
    with open('eval_result.txt', 'w') as fp:
        fp.write(f'Total samples: {len(preds)}\n')
        fp.write(f'Total correct: {len(correct)}\n')
        fp.write(f'Accuracy: {acc}\n')
        fp.write(f'F1-score: {f1}\n')
        fp.write(f'Label 0 -> predict 1: {len(l0_p1)}    percent: {len(l0_p1)/len(preds)}\n')
        fp.write(f'Label 1 -> predict 0: {len(l1_p0)}    percent: {len(l1_p0)/len(preds)}\n')
        fp.write(f'l0p1: {len(l0_p1)}\n')
        fp.write(f'l0p0: {len(l0_p0)}\n')
        fp.write(f'l1p1: {len(l1_p1)}\n')
        fp.write(f'l1p0: {len(l1_p0)}\n')
        count_re = 0
        count_b0 = 0
        count_b1 = 0
        count_co = 0
        for i in l0_p1:
            if i in l1_p0:
                count_re += 1
            elif i in l1_p1:
                count_b1 += 1
        for i in l0_p0:
            if i in l1_p1:
                count_co += 1
            elif i in l1_p0:
                count_b0 += 1
        fp.write(f'Samples predicted both Vulnerable: {count_b1}, percent: {count_b1/(len(preds)/2)}\n')
        fp.write(f'Samples predicted both Non-Vulerable: {count_b0}, percent: {count_b0/(len(preds)/2)}\n')
        fp.write(f'Samples precicted Reversed: {count_re}, precent: {(count_re)/(len(preds)/2)}\n')
        fp.write(f'Samples precicted Correct: {count_co}, precent: {(count_co)/(len(preds)/2)}\n')
    with open('l0_p1.txt', 'w') as fp:
        fp.write('\n'.join(list(map(str, l0_p1))))
    with open('l1_p0.txt', 'w') as fp:
        fp.write('\n'.join(list(map(str, l1_p0))))
    with open('correct.txt', 'w') as fp:
        fp.write('\n'.join(list(map(str, correct))))


def eval_diff():
    with open('./prediction.txt', 'r') as fp:
        preds = fp.readlines()
    labels = []
    logits = []
    indexs = []
    for pred in preds:
        idx, label, logit, probe = pred.split('\t')
        idx = int(idx.split(':')[-1].strip())
        label = int(label.split(':')[-1].strip())
        logit = int(logit.split(':')[-1].strip())
        labels.append(label)
        logits.append(logit)
        indexs.append(idx)
    acc = accuracy_score(labels, logits)
    f1 = f1_score(labels, logits)
    info = {}
    for pred, label, idx in zip(logits, labels, indexs):
        if idx not in info:
            info[idx] = [-1, -1]
        info[idx][label] = pred
    all_1, all_0, reverse, correct = 0, 0, 0, 0
    count = 0
    for idx, item in info.items():
        s_pred, v_pred = item
        if s_pred == -1 or v_pred == -1:
            continue
        if s_pred == v_pred == 1:
            all_1 += 1
        elif s_pred == v_pred == 0:
            all_0 += 1
        elif s_pred == 1 and v_pred == 0:
            reverse += 1
        elif s_pred == 0 and v_pred == 1:
            correct += 1
        count += 1
    with open('eval_result.txt', 'w') as fp:
        fp.write(f'Total samples: {len(preds)}\n')
        fp.write(f'Accuracy: {acc}\n')
        fp.write(f'F1-score: {f1}\n')
        fp.write(f'Samples predicted both Vulnerable: {all_1}, percent: {all_1/count*100}\n')
        fp.write(f'Samples predicted both Non-Vulerable: {all_0}, percent: {all_0/count*100}\n')
        fp.write(f'Samples precicted Reversed: {reverse}, precent: {reverse/count*100}\n')
        fp.write(f'Samples precicted Correct: {correct}, precent: {correct/count*100}\n')


def calc_diff_with_names():
    with open('./prediction.txt', 'r') as fp:
        preds = fp.readlines()
    with open('./names.json', 'r') as fp:
        names = json.load(fp)
    name_list = names['name_list']
    idx_list = [name.split('_')[0] for name in name_list]
    info = {}
    for pred in preds:
        idx, label, logit, probe, frm, value = pred.split('\t')
        idx = idx.split(':')[-1].strip()
        label = int(label.split(':')[-1].strip())
        logit = int(logit.split(':')[-1].strip())
        if idx not in idx_list:
            continue
        if idx not in info:
            info[idx] = [-1, -1]
        info[idx][label] = logit
    all_1, all_0, reverse, correct = 0, 0, 0, 0
    count = 0
    for idx, item in info.items():
        s_pred, v_pred = item
        if s_pred == -1 or v_pred == -1:
            continue
        if s_pred == v_pred == 1:
            all_1 += 1
        elif s_pred == v_pred == 0:
            all_0 += 1
        elif s_pred == 1 and v_pred == 0:
            reverse += 1
        elif s_pred == 0 and v_pred == 1:
            correct += 1
        count += 1
    print('All calc number: %d' % count)
    print('All vuln: %0.2f\tAll safe: %0.2f\tReversed: %0.2f\tCorrect: %0.2f' % ((all_1/count)*100, (all_0/count)*100, (reverse/count)*100, (correct/count)*100))


if __name__ == '__main__':
    # main()
    eval_diff()
    # calc_diff_with_names()
