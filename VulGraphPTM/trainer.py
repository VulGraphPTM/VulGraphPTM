import copy
import json
from dgl import DGLGraph
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from utils import debug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False, with_tokens=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            if with_tokens:
                graph, targets, input_ids, node_ids_map = data_iter(with_tokens=True)
            else:
                graph, targets = data_iter()
            # graph, targets, input_ids = data_iter()
            targets = targets.cuda()
            if with_tokens:
                predictions = model(graph, input_ids=input_ids, cuda=True, device=device, node_ids_map=node_ids_map)
            else:
                predictions = model(graph, cuda=True, device=device)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter, output_ggnn, calc_diff=False, with_tokens=False, output_attention=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        all_ggnn_sum = []
        all_node_embeds = []
        all_names = []
        all_probes = []
        for _ in range(num_batches):
            if with_tokens:
                graph, targets, input_ids, names, node_ids_map = data_iter(ret_name=True, with_tokens=True)
            else:
                graph, targets, names = data_iter(ret_name=True)
            # graph, targets, input_ids, names = data_iter(ret_name=True)
            all_names += names
            targets = targets.cuda()
            if output_ggnn:
                predictions, ggnn_sum = model(graph, cuda=True, device=device, output_ggnn=output_ggnn)
                # predictions, node_embeds, ggnn_sum = model(graph, cuda=True, device=device, output_ggnn=output_ggnn)
            else:
                if with_tokens:
                    if output_attention:
                        predictions, gate = model(graph, input_ids=input_ids, cuda=True, device=device, output_ggnn=output_ggnn, node_ids_map=node_ids_map, output_attention=True)
                    else:
                        predictions = model(graph, input_ids=input_ids, cuda=True, device=device, output_ggnn=output_ggnn, node_ids_map=node_ids_map)
                else:
                    if output_attention:
                        predictions, gate = model(graph, cuda=True, device=device, output_ggnn=output_ggnn, output_attention=True)
                    else:
                        predictions = model(graph, cuda=True, device=device, output_ggnn=output_ggnn)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
                all_probes.extend(np.amax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
                all_probes.extend(predictions.numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            if output_ggnn:
                ggnn_sum = ggnn_sum.detach().cpu().numpy().tolist()
                # ggnn_sum = [g.detach().cpu().numpy().tolist() for g in ggnn_sum]
                # node_embeds = [g.detach().cpu().numpy().tolist() for g in node_embeds]
                all_ggnn_sum += ggnn_sum
                # all_node_embeds += node_embeds
        # save to prediction.txt
        print(len(all_names), len(all_targets), len(all_predictions), len(all_probes))
        assert len(all_names) == len(all_targets) == len(all_predictions) == len(all_probes)
        items = []
        for name, target, pred, prob in zip(all_names, all_targets, all_predictions, all_probes):
            items.append([name, 1 if target > 0.5 else 0, pred, prob])
        items.sort(key=lambda x: x[0])
        with open('prediction.txt', 'w') as fp:
            for item in items:
                fp.write(f'name: {item[0]}\tlabel: {item[1]}\tpred: {item[2]}\tprobe: {item[3]}\n')
        # model.train()
        if output_ggnn:
            # assert len(all_targets) == len(all_ggnn_sum) == len(all_node_embeds) == len(all_names)
            assert len(all_targets) == len(all_ggnn_sum) == len(all_names)
            all_items = []
            idx = 0
            # for (target, ggnn_sum, node_embed, name) in zip(all_targets, all_ggnn_sum, all_node_embeds, all_names):
            for (target, ggnn_sum, name) in zip(all_targets, all_ggnn_sum, all_names):
                item = {
                    'graph_feature': ggnn_sum,
                    # 'node_features':node_embed,
                    'target': target,
                    'file_name': name
                }
                idx += 1
                all_items.append(item)
            assert len(all_items) == len(all_targets)
            with open('./after_ggnn/ggnn_sum.json', 'w') as fp:
                json.dump(all_items, fp)
        if calc_diff:
            assert len(all_predictions) == len(all_names)
            info = {}
            for pred, name in zip(all_predictions, all_names):
                idx = name.split('_')[0]
                # if idx.isdigit():
                #     idx = name.split('_')[0] + '_' + name.split('_')[1]
                if idx not in info:
                    info[idx] = [-1, -1]
                if name.endswith('.c'):
                    label = int(name.split('_')[-1].replace('.c', ''))
                else:
                    label = int(name.split('_')[-1].replace('.txt', ''))
                info[idx][label] = pred
            with open('./pred_info.json', 'w') as fp:
                json.dump(info, fp)
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
            debug('All calc number: %d' % count)
            debug('All vuln: %0.2f\tAll safe: %0.2f\tReversed: %0.2f\tCorrect: %0.2f' % ((all_1/count)*100, (all_0/count)*100, (reverse/count)*100, (correct/count)*100))
        if output_attention:
            print(gate.size())
        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=5, output_ggnn=False, with_tokens=False):
    # print('n_gpu: ' + str(n_gpu))
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    try:
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            if with_tokens:
                graph, targets, input_ids, node_ids_map = dataset.get_next_train_batch(with_tokens=with_tokens)
            else:
                graph, targets = dataset.get_next_train_batch()
            targets = targets.cuda()
            if with_tokens:
                predictions = model(graph, input_ids=input_ids, cuda=True, device=device, output_ggnn=output_ggnn, node_ids_map=node_ids_map)
            else:
                predictions = model(graph, cuda=True, device=device, output_ggnn=output_ggnn)
            batch_loss = loss_function(predictions, targets)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            if step_count % dev_every == (dev_every - 1):
                valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_valid_batch(),
                                                     dataset.get_next_valid_batch, with_tokens=with_tokens)
                if valid_f1 > best_f1:
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    patience_counter += 1
                debug('\nStep %d\t\tTrain Loss: %10.3f\tValid Loss: %10.3f\tf1: %5.2f\tPatience %d' % (
                    step_count, np.mean(train_losses).item(), valid_loss, valid_f1, patience_counter))
                debug('=' * 100)
                train_losses = []
                if patience_counter == max_patience:
                    # last_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path + '-last-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch, output_ggnn=output_ggnn, with_tokens=with_tokens)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)


def test(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=5, output_ggnn=False, calc_diff=False, with_tokens=False, output_attention=False):
    # _save_file = open(save_path + '-model.bin', 'rb')
    # model = torch.load(save_path + '-model.bin')
    # _save_file.close()
    # print('n_gpu: ' + str(n_gpu))
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(save_path + '-model.bin'))
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch, output_ggnn, calc_diff, with_tokens, output_attention)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)
