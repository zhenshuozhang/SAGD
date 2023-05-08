import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import load, load_coauthor
from torch_geometric.seed import seed_everything
import dgl
from dgl.data import register_data_args, load_data
import dgl.function as fn
import random
import copy
from models import Classifier, SAGD
import os
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False # may slow down the program
    torch.backends.cudnn.determinstic = True

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels.argmax(1))
        return correct.item() * 1.0 / len(labels.argmax(1))

def main(args, run):
    if args.seed:
        set_random_seed(run)
    # load and preprocess dataset
    # data = load_data(args)
    cuda = True
    free_gpu_id = 0
    torch.cuda.set_device(int(free_gpu_id))
    device = f'cuda:{free_gpu_id}' if torch.cuda.is_available() else 'cpu'
    adj, features, labels, idx_train, idx_val, idx_test = load_coauthor(args.dataset_name)

    train_val_ratio = 0.2

    idx_train_val = random.sample(list(np.arange(features.shape[0])), int(train_val_ratio * features.shape[0]))
    remain_num = len(idx_train_val)
    idx_train = idx_train_val[remain_num//2:]
    idx_val = idx_train_val[:remain_num//2]
    idx_test = list(set(np.arange(features.shape[0])) - set(idx_train_val))

    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(features)
    g.ndata['label'] = torch.LongTensor(labels)
    n_nodes = features.shape[0]

    mask = ['train_mask', 'test_mask', 'val_mask']
    for i, idx in enumerate([idx_train, idx_test, idx_val]):
        temp_mask = torch.zeros(g.num_nodes())
        temp_mask[idx] = 1
        g.ndata[mask[i]] = temp_mask.bool()

    g, labels, train_idx, val_idx, test_idx, features = map(
        lambda x: x.to(free_gpu_id), (g, g.ndata['label'], g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'], g.ndata['feat'])
    )
    print(g)
    print(f"Total edges before adding self-loop {g.number_of_edges()}")
    g = g.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {g.number_of_edges()}")
    in_feats = g.ndata['feat'].shape[1]
    n_classes = labels.shape[1]
    n_edges = g.num_edges()

    g = g.to(free_gpu_id)
    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    norm = norm.to(device).unsqueeze(1)
    features_hop_list = []
    features_hop_list.append(features.unsqueeze(0))
    for i in range(args.khop):
        feat = features_hop_list[-1].squeeze(0) * norm
        g.ndata['h'] = feat
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        temp_rep = g.ndata.pop('h') * norm
        features_hop_list.append(temp_rep.unsqueeze(0))
    ori_feats = torch.cat(features_hop_list[1:]).to('cpu')

    degree_list = g.in_degree(range(n_nodes)).to('cpu')
    nei_d_list = []
    if args.d_pred > 0:
        for i in tqdm(range(n_nodes)):
            nei_d_list.append(g.in_degree(g.successors(i)).to(torch.float).mean().item())
        nei_d_list = torch.tensor(nei_d_list)
        r_list = degree_list / nei_d_list
    
    hop_labels = []
    khop = args.khop
    #hop_labels.append(torch.zeros(n_nodes))
    for i in range(khop):
        hop_labels.append(torch.ones(n_nodes)*i)
    hop_labels = torch.cat(hop_labels).long()
    # create model
    model = SAGD(in_feats, args.n_hidden, args.n_layers, 'prelu', args.khop, args.dropout, 'cat', args.lam_train+args.mix_train)
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    avg_time = 0
    dur = []
    n_nodes = g.num_nodes()
    khop = args.khop


    tag = str(int(np.random.random() * 10000000000))
    if args.ad:
        epochs  = int(args.epochs/args.adm)
    else:
        epochs = args.epochs
    loop=tqdm(range(epochs))
    for epoch in loop:

        n_nodes = features.shape[0]
        sample_num = int(n_nodes/khop)
        sample = np.random.choice(range(n_nodes), sample_num, replace=False)
        idx = np.random.permutation(n_nodes)
        # aug pos features
        aug_features = aug_feature_dropout(features, args.drop_prob).cuda()
        g.ndata['aug_feat'] = aug_features
        aug_features_hop_list = []
        aug_features_hop_list.append(aug_features.unsqueeze(0))
        for i in range(khop):
            feat = aug_features_hop_list[-1].squeeze(0) * norm
            g.ndata['h'] = feat
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            temp_rep = g.ndata.pop('h') * norm
            aug_features_hop_list.append(temp_rep.unsqueeze(0))
        aug_feats_all = torch.cat(aug_features_hop_list[1:]).to('cpu')
        # aug neg features
        shuf_features = aug_features[idx]
        g.ndata['shuf_feat'] = shuf_features
        shuf_features_hop_list = []
        shuf_features_hop_list.append(shuf_features.unsqueeze(0))
        for i in range(khop):
            feat = shuf_features_hop_list[-1].squeeze(0) * norm
            g.ndata['h'] = feat
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            temp_rep = g.ndata.pop('h') * norm
            shuf_features_hop_list.append(temp_rep.unsqueeze(0))
        shuf_feats_all = torch.cat(shuf_features_hop_list[1:]).to('cpu')
        t0 = time.time()

        if args.sample_hop:
            aug_feats = aug_feats_all[:, sample, :]
            shuf_feats = shuf_feats_all[:, sample, :]
        else:
            aug_feats = aug_feats_all[-1].unsqueeze(0)
            shuf_feats = shuf_feats_all[-1].unsqueeze(0)
        n_nodes = aug_feats.shape[1]*aug_feats.shape[0]
        model.train()
        if args.sample_hop:
            hop_labels = []
            #hop_labels.append(torch.zeros(n_nodes))
            for i in range(khop):
                hop_labels.append(torch.ones(sample_num)*i)
            hop_labels = torch.cat(hop_labels).long()
        if epoch >= 3:
            t0 = time.time()

        optimizer.zero_grad()

        lbl_1 = torch.ones(1, n_nodes)
        lbl_2 = torch.zeros(1, n_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
        if torch.cuda.is_available():
            shuf_feats = shuf_feats.cuda()
            aug_feats = aug_feats.cuda()
            lbl = lbl.cuda()
        if args.ad:
            step_size = args.step_size
            m = args.adm
            perturb_shape = aug_feats.shape
            perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
            perturb.requires_grad_()
            shuf_perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
            shuf_perturb.requires_grad_()
            
            aug_feats += perturb
            shuf_feats += shuf_perturb
            if args.lam_train:
                logits_opt_h = model.opt_h(aug_feats, shuf_feats)
                loss_opt_h = b_xent(logits_opt_h, lbl)

                logits_opt_lam = model.opt_lam(aug_feats, shuf_feats)
                loss_opt_lam = -b_xent(logits_opt_lam, lbl)*0.01
            else:
                logits, d_logits_1, d_logits_2, hop_logits_1, hop_logits_2 = model(aug_feats, shuf_feats)
                loss_disc = b_xent(logits, lbl)

            for i in range(m-1):
                if args.lam_train:
                    loss_opt_h.backward()
                    loss_opt_lam.backward()
                else:
                    loss_disc.backward()
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
                shuf_perturb_data = shuf_perturb.detach() + step_size * torch.sign(shuf_perturb.grad.detach())
                shuf_perturb.data = shuf_perturb_data.data
                shuf_perturb.grad[:] = 0

                aug_feats += perturb
                shuf_feats += shuf_perturb
                if args.lam_train:
                    logits_opt_h = model.opt_h(aug_feats, shuf_feats)
                    loss_opt_h = b_xent(logits_opt_h, lbl)

                    logits_opt_lam = model.opt_lam(aug_feats, shuf_feats)
                    loss_opt_lam = -b_xent(logits_opt_lam, lbl)*0.1
                else:
                    logits, d_logits_1, d_logits_2, hop_logits_1, hop_logits_2 = model(aug_feats, shuf_feats)
                    loss_disc = b_xent(logits, lbl)

        else:
            logits, d_logits_1, d_logits_2, hop_logits_1, hop_logits_2 = model(aug_feats, shuf_feats) # return group logits and hop logits
            loss_disc = b_xent(logits, lbl)
            loss_mlp = model.enc.fc.weight.data.T.std().mean()
        # import pdb; pdb.set_trace()
        logits, d_logits_1, d_logits_2, hop_logits_1, hop_logits_2 = model(aug_feats, shuf_feats) # return group logits and hop logits
        loss_disc = b_xent(logits, lbl)
        if args.lam_train and not args.ad:
            logits_opt_h = model.opt_h(aug_feats, shuf_feats)
            loss_opt_h = b_xent(logits_opt_h, lbl)

            logits_opt_lam = model.opt_lam(aug_feats, shuf_feats)
            loss_opt_lam = -b_xent(logits_opt_lam, lbl)*0.01
            loss = loss_opt_h
        if args.lam_train:
            loss = loss_opt_h
        else:
            loss = loss_disc
        if args.hop_pred > 0:
            #hop_logits_1 = model.hop_pred(aug_feats.view(-1, khop), sp_adj, sparse=True)
            #hop_logits_2 = model.hop_pred(shuf_feats.view(-1, khop), sp_adj, sparse=True)
            loss_hop_1 = xent(hop_logits_1, hop_labels.cuda())
            loss_hop_2 = xent(hop_logits_2, hop_labels.cuda())
            loss_hop = loss_hop_1 + loss_hop_2
            loss += loss_hop * args.hop_pred

        if args.d_pred > 0:
            df = (aug_features_hop_list[-1] - aug_features_hop_list[1]).squeeze(0).cuda()
            shuf_df = (shuf_features_hop_list[-1] - shuf_features_hop_list[1]).squeeze(0).cuda()
            #d_logits_1 = model.d_pred(df[sample], sp_adj, sparse=True)
            #d_logits_2 = model.d_pred(shuf_df[sample], sp_adj, sparse=True)
            #d_logits_1 = model.d_pred2(aug_feats, sp_adj, sparse=True)
            #d_logits_2 = model.d_pred2(shuf_feats, sp_adj, sparse=True)
            #d_labels = torch.cat(((r_list[sample]>=1).to(torch.long), (r_list[sample]>=1).to(torch.long))).cuda()
            d_labels =(r_list[sample]>=1).to(torch.long).cuda()
            loss_d_1 = xent(d_logits_1, d_labels)
            loss_d_2 = xent(d_logits_2, d_labels)
            loss_d = loss_d_1 + loss_d_2
            #loss_d = loss_d_1
            loss += loss_d * args.d_pred
        if args.lam_train:
            loss_opt_h.backward()
            loss_opt_lam.backward()
        else:
            loss.backward()
        optimizer.step()

        comp_time = time.time() - t0
        # print('{} seconds'.format(comp_time))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/best_model' + tag + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        avg_time += comp_time
        counts += 1
        loop.set_postfix_str('In epoch {}, loss: {:.3f}, best: {})'.format(epoch, loss, best))

    # create classifier model
    epoch_time = np.mean(dur)
    if args.ad:
        epoch_time /= args.adm
    print('per epoch {}'.format(epoch_time))
    classifier = Classifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=args.cls_lr,
                                            weight_decay=args.weight_decay)

    # train classifier
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('pkl/best_model' + tag + '.pkl'))

    infer_feats = ori_feats[-1].cuda()
    inf_start = time.time()

    for i in range(args.diff_infer - args.khop):
        feat = features_hop_list[-1].squeeze(0) * norm
        g.ndata['h'] = feat
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        temp_rep = g.ndata.pop('h') * norm
        features_hop_list.append(temp_rep.unsqueeze(0))
    if args.diff_infer > args.khop:
        infer_feats += features_hop_list[-1].squeeze(0)

    or_embeds = model.embed(infer_feats).detach()

    embeds = or_embeds


    dur = []
    for epoch in range(args.cls_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[g.ndata['train_mask']], labels[g.ndata['train_mask']].argmax(1))
        loss.backward()
        classifier_optimizer.step()
        
        if epoch >= 3:
            dur.append(time.time() - t0)

    acc = evaluate(classifier, embeds, labels, g.ndata['test_mask'])
    print("Test Accuracy {:.4f} in run {}".format(acc, run))

    return acc

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='sagd')
    register_data_args(parser)

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")

    parser.add_argument("--n-hidden", type=int, default=1024,
                        help="number of hidden gcn units")
    parser.add_argument("--proj_layers", type=int, default=1,
                        help="number of project linear layers")

    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--n_trails", type=int, default=5,
                        help="number of trails")
    parser.add_argument("--gnn_encoder", type=str, default='gcn',
                        help="choice of gnn encoder")
    parser.add_argument("--num_hop", type=int, default=10,
                        help="number of k for sgc")
    parser.add_argument('--data_root_dir', type=str, default='default',
                           help="dir_path for saving graph data. Note that this model use DGL loader so do not misx up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")
    parser.add_argument("--pretrain_path", type=str, default='None',
                        help="path for pretrained node features")
    parser.add_argument('--dataset_name', type=str, default='photo',
                        help='Dataset name: computer, photo')
    
    parser.add_argument('--cls_epochs', type=int, default=2000, help='classifier epochs')
    parser.add_argument('--cls_lr', type=float, default=0.05, help='classifier epochs')
    parser.add_argument('--batch_size', type=int, default=0, help='batch_size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--lr', type=float, default=5e-4 , help='Patience')
    parser.add_argument('--l2_coef', type=float, default=0.0, help='l2 coef')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--drop_prob', type=float, default=0.0, help='Tau value')
    parser.add_argument('--hid', type=int, default=1024, help='Top-K value')
    parser.add_argument('--sparse', action='store_true', help='Whether to use sparse tensors')
    parser.add_argument('--runs', type=int, default=10, help='number of trails')
    parser.add_argument('--khop', type=int, default=2, help='khop')
    parser.add_argument('--m', type=str, default='mlp', help='model')
    parser.add_argument('--op', type=str, default='cat', help='opreation')
    parser.add_argument('--pre_pool', type=int, default=0, help='augumentation pool')
    parser.add_argument('--hop_pred', type=float, default=0, help='hop prediction weight')
    parser.add_argument('--d_pred', type=float, default=0, help='degree prediction weight')
    parser.add_argument('--n_layers', type=int, default=1, help='encoder model layer')
    parser.add_argument('--lam_train', action='store_true', default=True, help='lambda weighted train')
    parser.add_argument('--mix_train', action='store_true', default=False, help='mixed lambda weighted train')
    parser.add_argument('--lam_reg', type=float, default=0, help='mixed lambda weighted train')
    parser.add_argument('--lam_infer', action='store_true', help='lambda weighted inference')
    parser.add_argument('--sample_hop', action='store_true', default=True, help='sample hop in train')
    parser.add_argument('--rank_train', action='store_true', help='sample hop in train')
    parser.add_argument('--diff_infer', type=int, default=0, help='diffusion inference')
    parser.add_argument('--seed', action='store_true', help='keep random seed')
    parser.add_argument('--ad', action='store_true', help='combine with adversarial training', default=False)
    parser.add_argument('--step_size', type=float, default=5e-5)
    parser.add_argument('--adm', type=int, default=3)
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    seed = False

    if seed:
        seed = args.seed
        seed_everything(seed)
        print('seed_number:' + str(seed))

    accs = []
    for i in range(args.n_trails):
        accs.append(main(args, i))
    print('mean accuracy:' + str(np.array(accs).mean()) + "±" + str(np.array(accs).std()))
    

    with open('gslog_{}.txt'.format(args.dataset_name), 'a') as f:
        f.write(str(args))
        f.write('\n' + str(np.mean(accs)) + '\n')
        f.write(str(np.std(accs)) + '\n')
