import os
import pickle
import time
import json
import argparse
from objects.cpg import Cpg
from objects.fpg import FPG
from os.path import join, exists
from objects.w2vmodel import generate_w2vModel, load_w2vModel
from tqdm import tqdm


def program_functions_to_graphs(cpg: Cpg, group):
    fpg_list = []
    vul_count = 0
    non_vul_count = 0
    none_count = 0
    for method in cpg.methods:
        label = method.label
        code = method.code
        if not code:
            none_count += 1
        assert label == 0 or label == 1
        if label == 1:
            vul_count += 1
        else:
            non_vul_count += 1
        fpg = FPG(method.name, method, label, code=code)
        if len(fpg.node_list) < 5:
            continue
        fpg_list.append(fpg)
    print(f"None code: {none_count}")
    if not exists('./cache'):
        os.mkdir('./cache')
    save_path = join('./cache', f"fpg_list_{group}.pkl")
    with open(save_path, "wb") as fp:
        pickle.dump(fpg_list, fp)
    time.sleep(0.5)
    print(f"vul functions number: {vul_count}")
    print(f"non-vul functions number: {non_vul_count}")
    return fpg_list


def program_functions_to_graphs_with_load(group):
    target = join('./cache', f"fpg_list_{group}.pkl")
    with open(target, 'rb') as fp:
        fpg_list = pickle.load(fp)
    return fpg_list


def graph_to_dataset_new(
        cpg: Cpg,
        group: int,
        w2v_path: str,
        save_path: str,
        args
):
    # generate slice/function program slice
    if args.gen_graph:
        g_list = program_functions_to_graphs(cpg, group)
    else:
        try:
            g_list = program_functions_to_graphs_with_load(group)
        except FileNotFoundError:
            return
    # generate corpus
    # convert slice/function program graph to model input dataset
    if args.g2dataset:
        # load embedding model
        wv = load_w2vModel(w2v_path)
        # embed and save
        graphs = []
        for graph in tqdm(g_list, desc="convert graphs to dataset..."):
            gInput = graph.embed(args.node_dim, wv)
            if gInput:
                graphs.append(gInput)
        with open(save_path, 'w') as fp:
            json.dump(graphs, fp)


def arg_parse():
    parser = argparse.ArgumentParser(description="Data pre-processing arguments")
    parser.add_argument('--dataset', dest='dataset', help='Dataset to process')
    parser.add_argument('--portion', dest='portion', choices=['train', 'valid', 'test', 'nan'])
    parser.add_argument('--group', dest='group', type=int, help='Pre-processing group of selected dataset')
    parser.add_argument(
        '--node-dim',
        dest='node_dim',
        type=int,
        help='Nodes dim of each slice/function graph; preprocessing program would cut or pad the nodes matrix '
             'to meet the length.'
    )
    parser.add_argument(
        '--vul-ratio',
        dest='vul_ratio',
        type=int,
        help='Ratio of non-vulnerable data to vulnerable data. '
             'Default is 1:1, when setting to 3, it means vul-data: non-vul-data is 1:3'
    )
    parser.add_argument(
        '--gen-graph',
        dest='gen_graph',
        action='store_const', const=True,
        help='Generate slice/function program graphs from source dataset'
    )
    parser.add_argument(
        '--with-load',
        dest='with_load',
        action='store_const', const=True,
        help='Load existing slice/function graph from spgs/fpgs dir'
    )
    parser.add_argument(
        '--g2dataset',
        dest='g2dataset',
        action='store_const', const=True,
        help='Convert slice/function program graphs to model input dataset'
    )
    parser.add_argument(
        '--w2v-dir',
        dest='w2v_dir',
        help='Directory to save embedding models. Default dir is `./input/w2v`'
    )
    parser.add_argument(
        '--save-dir',
        dest='dataset_dir',
        help='Directory to save input dataset'
    )
    parser.add_argument(
        '--src_path',
        dest='src_path',
        help='Directory to save input dataset'
    )

    parser.set_defaults(
        dataset='devign',
        portion='test',
        group=0,
        node_dim=130,
        vul_ratio=1,
        gen_graph=False,
        with_load=False,
        g2dataset=False,
        w2v_dir="./cache/",
        save_dir="/data2/zhujh/reveal_data/",
        src_path=None
    )

    return parser.parse_args()


def graph_process():
    args = arg_parse()
    print(args)
    cpg_path = f'/data2/zhujh/joern-files/cache/results_{args.dataset}/{args.portion}_group{args.group}'
    if not exists('./cache'):
        os.mkdir('./cache')
    start_time = time.time()
    cpg = Cpg(cpg_path, args.src_path)
    wv_path = join(args.w2v_dir, f"w2v_model_{args.dataset}.model")
    save_path = join(args.save_dir, f"{args.dataset}_{args.portion}_group{args.group}-full_graph.json")
    graph_to_dataset_new(cpg, args.group, wv_path, save_path, args)
    end_time = time.time()
    print(f"Total process time: {end_time - start_time} s.")


def train_wv_model():
    corpus_list = [
        '/data/zhujh/dataset/cleaned/test.jsonl',
        '/data/zhujh/dataset/cleaned/valid.jsonl',
        '/data/zhujh/dataset/cleaned/train.jsonl'
    ]
    wv_path = join('./cache', "w2v_model_balance.model")
    generate_w2vModel(corpus_list, wv_path, vector_dim=130)


if __name__ == "__main__":
    # train_wv_model()
    graph_process()
