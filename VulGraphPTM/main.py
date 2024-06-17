import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam, AdamW

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum, RelGraphConvModel, BertWithGAP
from modules.ample import AMPLEModel
from trainer import train, test
from utils import tally_param, debug
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

import math
import torch.optim
from torch.optim import Optimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_id = torch.cuda.current_device()
print("Current GPU ID:", device_id)


if __name__ == '__main__':
    torch.manual_seed(12345)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn', 'rgcn', 'ample', 'bertgam'], default='devign')
    parser.add_argument('--action', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)
    parser.add_argument('--save_after_ggnn', action='store_true')
    parser.add_argument('--calc_diff', action='store_true')
    parser.add_argument('--output_attention', action='store_true')

    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    args = parser.parse_args()

    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size
    
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    ptm_model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                                 ignore_mismatched_sizes=True)
    # config = None
    # tokenizer = None
    # ptm_model = None

    # model_dir = os.path.join('models', args.dataset)
    model_dir = os.path.join('models', 'VulFixed')
    if args.action == 'train':
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    input_dir = args.input_dir
    print(args)
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if False and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        if args.action == 'train':
            train_path = os.path.join(input_dir, f'{args.dataset}_train-full_graph.json')
            valid_path = os.path.join(input_dir, f'{args.dataset}_valid-full_graph.json')
            test_path = os.path.join(input_dir, f'{args.dataset}_test-full_graph.json')
        else:
            test_path = os.path.join(input_dir, f'{args.dataset}-full_graph.json')
            train_path = None
            valid_path = None
        add_self_loop = False
        batch_type = 'lib'
        if args.model_type in ['devign', 'ample']:
            batch_type = 'origin'
        dataset = DataSet(train_src=train_path,
                          valid_src=valid_path,
                          test_src=test_path,
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag,
                          add_self_loop=add_self_loop,
                          batch_type=batch_type,
                          tokenizer=tokenizer)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    print('edge types: ', dataset.max_edge_type)
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'rgcn':
        model = RelGraphConvModel(input_dim=dataset.feature_size, h_dim=args.graph_embed_size, out_dim=args.graph_embed_size,
                                  num_relations=dataset.max_edge_type, num_hidden_layers=1)
    elif args.model_type == 'ample':
        model = AMPLEModel(input_dim=dataset.feature_size, output_dim=dataset.feature_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'bertgam':
        model = BertWithGAP(input_dim=dataset.feature_size, h_dim=args.graph_embed_size, out_dim=args.graph_embed_size,
                            num_relations=dataset.max_edge_type, encoder=ptm_model, tokenizer=tokenizer, config=config, num_hidden_layers=1)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.001)
    
    if args.model_type == 'rgcn':
        if args.action == 'train':
            train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/RGCN-GAP', max_patience=50, log_every=None)
        else:
            test(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/RGCN-GAP', max_patience=50, log_every=None, output_ggnn=args.save_after_ggnn, calc_diff=args.calc_diff, output_attention=args.output_attention)
    elif args.model_type == 'ample':
        if args.action == 'train':
            train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/AMPLE', max_patience=50, log_every=None)
        else:
            test(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/AMPLE', max_patience=50, log_every=None, output_ggnn=args.save_after_ggnn, calc_diff=args.calc_diff)
    elif args.model_type == 'bertgam':
        if args.action == 'train':
            train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/BertwithGGAP', max_patience=50, log_every=None, with_tokens=True)
        else:
            test(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_dir + '/BertwithGGAP', max_patience=50, log_every=None, output_ggnn=args.save_after_ggnn, calc_diff=args.calc_diff, with_tokens=True)
    else:
        if args.action == 'train':
            train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
                loss_function=loss_function, optimizer=optim,
                save_path=model_dir + '/DevignModel', max_patience=50, log_every=None)
        else:
            test(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
                loss_function=loss_function, optimizer=optim,
                save_path=model_dir + '/DevignModel', max_patience=50, log_every=None, output_ggnn=args.save_after_ggnn, calc_diff=args.calc_diff)
