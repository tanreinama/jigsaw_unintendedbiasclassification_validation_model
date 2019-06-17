import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm import tqdm
from fastcache import clru_cache as lru_cache

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import *
from fastai.basic_data import DatasetType

from tokenvectorizer import TokenVectorizer
from makemodel import get_model

tqdm.pandas()

def read_competision_file(train=True):
    if train:
        df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
    else:
        df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
    return df

CRAWL_EMBEDDING_PATH = 'fasttext_crawl_withtag-300d-2M.bz2'
#CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_isolated_embedding(matrix, x_train, y_train_o):
    p_matrix = np.zeros((matrix.shape[1],))
    n_matrix = np.zeros((matrix.shape[1],))
    p_count = 0
    n_count = 0
    for i in tqdm(range(x_train.shape[0])):
        for w in x_train[i]:
            if w > 0:
                if y_train_o[i] > 0:
                    p_matrix += matrix[w]
                    p_count += 1
                else:
                    n_matrix += matrix[w]
                    n_count += 1
    p_matrix /= p_count
    n_matrix /= n_count
    if len(args.model_file) > 0:
        with open('embedding_means.json' if '/' not in args.model_file else '/'.join(args.model_file.split('/')[:-1])+'/embedding_means.json','w') as pf:
            pf.write('{')
            pf.write('\"p_matrix\":')
            pf.write('[')
            pf.write(','.join(list(map(str,p_matrix))))
            pf.write(']')
            pf.write(',')
            pf.write('\"n_matrix\":')
            pf.write('[')
            pf.write(','.join(list(map(str,n_matrix))))
            pf.write(']')
            pf.write('}')
    p_embedding = matrix - p_matrix
    n_embedding = matrix - n_matrix
    return np.concatenate([p_embedding, n_embedding], axis=-1)

def make_temp():
    print('Read file.')
    train = read_competision_file(train=True)
    test = read_competision_file(train=False)

    if args.use_feats_url:
        train['num_http'] = train['comment_text'].apply(lambda x: str(x).count('http'))
        test['num_http'] = test['comment_text'].apply(lambda x: str(x).count('http'))
        train['num_words'] = train['comment_text'].apply(lambda x: len(str(x).split()))
        test['num_words'] = test['comment_text'].apply(lambda x: len(str(x).split()))
        train['http_per_words'] = np.int64((train['num_http'] / train['num_words']) * 1000)
        test['http_per_words'] = np.int64((test['num_http'] / test['num_words']) * 1000)
        x_feat_train = train[['num_http','num_words','http_per_words']].values
        x_feat_test = test[['num_http','num_words','http_per_words']].values
    else:
        x_feat_train = None
        x_feat_test = None

    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
    all_text = list(train['comment_text']) + list(test['comment_text'])

    y_train_o = (train['target'].values>=0.5).astype(np.int)

    if args.lemma_dict == '':
        lemma = {}
    else:
        lemma = args.lemma_dict
    if args.correct_dict == '':
        corre = {}
    else:
        with open(args.correct_dict, 'rb') as rf:
            corre = json.load(rf)
    tokenizer = TokenVectorizer(vector_dict=CRAWL_EMBEDDING_PATH, lemma_dict=lemma, spell_collector=corre)
    all_sent, crawl_matrix = tokenizer(all_text, args.num_words)

    print('n unknown words: ', len(tokenizer.unknown_words))
    with open('unknown_words.txt','w') as wf:
        wf.write('\n'.join(sorted(tokenizer.unknown_words)))

    x_train = all_sent[:len(train)]
    x_test = all_sent[len(train):]
    del test, x_test, all_text, all_sent

    embedding_matrix = make_isolated_embedding(crawl_matrix,x_train,y_train_o)

    with open(args.temporary_file, mode='wb') as f:
        pickle.dump((x_train, x_feat_train, y_train_o, y_aux_train, embedding_matrix), f)

def get_score():
    print('Make Train Features.')
    with open(args.temporary_file, 'rb') as f:
        x_train, x_feat_train, y_train_o, y_aux_train, embedding_matrix = pickle.load(f)

    def power_mean(series, p=-5):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # all, sub, s&t, !s&t, s&!t, !s&!t
    weight_factor = list(map(float,args.weight_factor.split(',')))
    identity_factor_1 = list(map(float,args.identity_factor_1.split(',')))
    identity_factor_2 = list(map(float,args.identity_factor_2.split(',')))
    model_factor = list(map(int,args.model_factor.split(',')))
    print('weight_factor =',weight_factor)
    print('identity_factor_1 = ',identity_factor_1)
    print('identity_factor_2 = ',identity_factor_2)
    print('model_factor = ',model_factor)
    train = read_competision_file(train=True)
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    index_subgroup, index_bpsn, index_bnsp = dict(), dict(), dict()
    for col in identity_columns:
        index_subgroup[col] = (train[col].fillna(0).values>=0.5).astype(bool)
        index_bpsn[col] = ( (( (train['target'].values<0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) ) + ( (( (train['target'].values>=0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values<0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) )
        index_bnsp[col] = ( (( (train['target'].values>=0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) ) + ( (( (train['target'].values<0.5).astype(bool).astype(np.int) + (train[col].fillna(0).values<0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool) )
    # Overall
    weights = np.ones((len(x_train),)) * weight_factor[0]
    # Subgroup
    weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) * weight_factor[1]
    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[2]
    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[3]
    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[4]
    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
       (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) * weight_factor[5]
    index_id1, index_id2 = dict(), dict()
    for col in identity_columns:
        index_id1[col] = (( (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) + (train['target'].values>=0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool)
        index_id2[col] = (( (train[col].fillna(0).values>=0.5).astype(bool).astype(np.int) + (train['target'].values<0.5).astype(bool).astype(np.int) ) > 1 ).astype(bool)
    for col,id1 in zip(identity_columns, identity_factor_1):
        weights[index_id1[col]] += id1
    for col,id2 in zip(identity_columns, identity_factor_2):
        weights[index_id2[col]] += id2

    loss_weight = 1.0 / weights.mean()

    aux_impact_factor = list(map(float,args.aux_impact_factor.split(',')))
    aux_identity_factor = list(map(float,args.aux_identity_factor.split(',')))
    print('aux_impact_factor =',aux_impact_factor)
    print('aux_identity_factor =',aux_identity_factor)

    weights_aux = np.ones((len(x_train),))
    weights_aux[(train['target'].values>=0.5).astype(np.int) + (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[0]
    weights_aux[(train['target'].values>=0.5).astype(np.int) + (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[1]
    weights_aux[(train['target'].values<0.5).astype(np.int) + (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[2]
    weights_aux[(train['target'].values<0.5).astype(np.int) + (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) > 1] = aux_identity_factor[3]

    y_train = np.vstack([y_train_o,weights,weights_aux]).T

    del train

    def custom_loss_aux(data, targets):
        ''' Define custom loss function for weighted BCE on 'target' column '''
        bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
        bce_loss_aux_1 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,1:2],targets[:,3:4])
        bce_loss_aux_2 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,2:3],targets[:,4:5])
        bce_loss_aux_3 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,3:4],targets[:,5:6])
        bce_loss_aux_4 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,4:5],targets[:,6:7])
        bce_loss_aux_5 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,5:6],targets[:,7:8])
        bce_loss_aux_6 = nn.BCEWithLogitsLoss(weight=targets[:,2:3])(data[:,6:7],targets[:,8:9])
        return (bce_loss_1 * loss_weight) + (bce_loss_aux_1 * aux_impact_factor[0]) + (bce_loss_aux_2 * aux_impact_factor[1]) + (bce_loss_aux_3 * aux_impact_factor[2]) + (bce_loss_aux_4 * aux_impact_factor[3]) + (bce_loss_aux_5 * aux_impact_factor[4]) + (bce_loss_aux_6 * aux_impact_factor[5])

    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    batch_size = args.batch_size
    lr = args.learning_ratio
    max_features = np.max(x_train)
    kf = KFold( n_splits=5, random_state=12, shuffle=True )
    final_epoch_score_cv = dict()
    final_fold_count = 0
    for fold_id, (big_index, small_index) in enumerate( kf.split( y_train ) ):
        final_fold_count += 1
        if args.minimize == 1:
            train_index, test_index = train_test_split(np.arange(len(y_train)), test_size=0.5, random_state=1234, shuffle=True)
        elif args.minimize == 2:
            train_index, test_index = train_test_split(np.arange(len(y_train)), test_size=0.666, random_state=1234, shuffle=True)
        elif args.minimize == 3:
            train_index, test_index = big_index[:25600], small_index[:12800]
        else:
            train_index, test_index = big_index, small_index

        if len(args.model_file) > 0:
            train_index = np.arange(len(x_train))

        if args.use_feats_url:
            x_train_train = np.hstack([x_feat_train[train_index],x_train[train_index]])
            x_train_test = np.hstack([x_feat_train[test_index],x_train[test_index]])
            feats_nums = x_feat_train.shape[1]
        else:
            x_train_train = x_train[train_index]
            x_train_test = x_train[test_index]
            feats_nums = 0

        x_train_torch = torch.tensor(x_train_train, dtype=torch.long)
        x_test_torch = torch.tensor(x_train_test, dtype=torch.long)
        y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train])[train_index], dtype=torch.float32)
        y_test_torch = torch.tensor(np.hstack([y_train, y_aux_train])[test_index], dtype=torch.float32)

        train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
        valid_dataset = data.TensorDataset(x_test_torch, y_test_torch)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        databunch = DataBunch(train_dl=train_loader,valid_dl=valid_loader)

        checkpoint_predictions = []
        weights = []
        seed_everything(args.random_seed + fold_id)
        num_units = list(map(int,args.num_units.split(',')))
        model = get_model(model_factor, num_units[0], num_units[1], embedding_matrix, max_features, y_aux_train.shape[-1], args.num_words, feats_nums)
        model = model.cuda(device=cuda)
        if args.optimizer == 'Nadam':
            from NadamLocal import Nadam
            learn = Learner(databunch,model,loss_func=custom_loss_aux,opt_func=Nadam)
        else:
            learn = Learner(databunch,model,loss_func=custom_loss_aux)
        all_test_preds = []
        checkpoint_weights = [2 ** epoch for epoch in range(args.num_epochs)]
        test_loader = valid_loader
        n = len(learn.data.train_dl)
        phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6**(i)))) for i in range(args.num_epochs)]
        sched = GeneralScheduler(learn, phases)
        learn.callbacks.append(sched)
        final_epoch_score = 0
        for global_epoch in range(args.num_epochs):
            print("Fold#",fold_id,"epoch#",global_epoch)
            learn.fit(1)
            if args.minimize < 2 or (args.minimize >= 2 and global_epoch == int(args.num_epochs-1)):
                test_preds = np.zeros((len(test_index), 7))
                for i, x_batch in enumerate(test_loader):
                    X = x_batch[0].cuda()
                    y_pred = sigmoid(learn.model(X).detach().cpu().numpy())
                    test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

                all_test_preds.append(test_preds)

                prediction_one = test_preds[:,0].flatten()
                checkpoint_predictions.append(prediction_one)

                weights.append(2 ** global_epoch)
                predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
                y_true = (y_train[test_index,0]).reshape((-1,)).astype(np.int)
                roc_sub, roc_bpsn, roc_bnsp = [], [], []
                roc_sub_one, roc_bpsn_one, roc_bnsp_one = [], [], []
                for col in identity_columns:
                    if args.vervose:
                        print("Subgroup#",col,":")
                        print(classification_report(y_true[index_subgroup[col][test_index]], (predictions[index_subgroup[col][test_index]] >= 0.5).astype(np.int)))
                    if args.minimize < 2:
                        roc_sub.append(roc_auc_score(y_true[index_subgroup[col][test_index]], predictions[index_subgroup[col][test_index]]))
                    roc_sub_one.append(roc_auc_score(y_true[index_subgroup[col][test_index]], prediction_one[index_subgroup[col][test_index]]))
                    if args.vervose:
                        print("BPSN#",col,":")
                        print(classification_report(y_true[index_bpsn[col][test_index]], (predictions[index_bpsn[col][test_index]] >= 0.5).astype(np.int)))
                    if args.minimize < 2:
                        roc_bpsn.append(roc_auc_score(y_true[index_bpsn[col][test_index]], predictions[index_bpsn[col][test_index]]))
                    roc_bpsn_one.append(roc_auc_score(y_true[index_bpsn[col][test_index]], prediction_one[index_bpsn[col][test_index]]))
                    if args.vervose:
                        print("BNSP#",col,":")
                        print(classification_report(y_true[index_bnsp[col][test_index]], (predictions[index_bnsp[col][test_index]] >= 0.5).astype(np.int)))
                    if args.minimize < 2:
                        roc_bnsp.append(roc_auc_score(y_true[index_bnsp[col][test_index]], predictions[index_bnsp[col][test_index]]))
                    roc_bnsp_one.append(roc_auc_score(y_true[index_bnsp[col][test_index]], prediction_one[index_bnsp[col][test_index]]))
                if args.minimize < 2:
                    roc_all = roc_auc_score(y_true, predictions)
                    pm_roc_sub = power_mean(roc_sub)
                    pm_roc_bpsn = power_mean(roc_bpsn)
                    pm_roc_bnsp = power_mean(roc_bnsp)
                    final_epoch_score = (roc_all+pm_roc_sub+pm_roc_bpsn+pm_roc_bnsp)/4
                roc_all_one = roc_auc_score(y_true, prediction_one)
                pm_roc_sub_one = power_mean(roc_sub_one)
                pm_roc_bpsn_one = power_mean(roc_bpsn_one)
                pm_roc_bnsp_one = power_mean(roc_bnsp_one)
                final_epoch_score_one = (roc_all_one+pm_roc_sub_one+pm_roc_bpsn_one+pm_roc_bnsp_one)/4
                if args.minimize >= 2:
                    return final_epoch_score_one
                if args.vervose:
                    print("roc_sub:",pm_roc_sub)
                    print("roc_bpsn:",pm_roc_bpsn)
                    print("roc_bnsp:",pm_roc_bnsp)
                    print("final score:",(roc_all+pm_roc_sub+pm_roc_bpsn+pm_roc_bnsp)/4)
                if global_epoch not in final_epoch_score_cv.keys():
                    final_epoch_score_cv[global_epoch] = []
                final_epoch_score_cv[global_epoch].append((final_epoch_score,final_epoch_score_one))
        if len(args.model_file) > 0:
            if args.model_file.endswith('.bz2'):
                model_file = args.model_file
            else:
                model_file = args.model_file + '.bz2'
            model_json_file = model_file[:-4]+'.json'
            model.save_model(model_file)
            with open(model_json_file,'w') as pf:
                pf.write('{')
                pf.write('\"model_factor\":['+','.join(list(map(str,model_factor)))+']')
                pf.write(',')
                pf.write('\"num_units\":['+','.join(list(map(str,num_units)))+']')
                pf.write(',')
                pf.write('\"num_aux_targets\":%d'%y_aux_train.shape[-1])
                pf.write(',')
                pf.write('\"feats_nums\":%d'%feats_nums)
                pf.write(',')
                pf.write('\"max_seq_len\":%d'%args.num_words)
                pf.write('}')
            break
        if args.minimize > 0:
            break
    return final_epoch_score_cv

if __name__=='__main__':
    import argparse
    ps = argparse.ArgumentParser( description='CV Test' )
    ps.add_argument( '--model_file', '-d', default='', help='Output Model File' )
    ps.add_argument( '--minimize', '-m', type=int, default=0, help='Minimize Test' )
    ps.add_argument( '--weight_factor', '-w', default='0.0,0.0,0.0,0.0,0.0,0.0', help='Weight factor' )
    ps.add_argument( '--identity_factor_1', '-1', default='0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0', help='Identity factor 1' )
    ps.add_argument( '--identity_factor_2', '-2', default='0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0', help='Identity factor 2' )
    ps.add_argument( '--model_factor', '-3', default='0,0,0,0,0,0', help='Model factor' )
    ps.add_argument( '--aux_impact_factor', '-4', default='1.0,1.0,1.0,1.0,1.0,1.0', help='Aux Target Impact factor' )
    ps.add_argument( '--aux_identity_factor', '-5', default='1.0,1.0,1.0,1.0', help='Aux Target Identity factor' )
    ps.add_argument( '--use_feats_url', '-f', action='store_true', help='Use URLs to Feats' )
    ps.add_argument( '--num_units', '-u', default='128,128', help='RNN Unit Size' )
    ps.add_argument( '--lemma_dict', '-a', default='lemma_dict-simbols.json', help='Lemma Charactor Dict JSON' )
    ps.add_argument( '--correct_dict', '-c', default='', help='Spel Collector Dict JSON' )
    ps.add_argument( '--num_words', '-n', type=int, default=220, help='Word Size' )
    ps.add_argument( '--batch_size', '-b', type=int, default=512, help='Batch Size' )
    ps.add_argument( '--num_epochs', '-e', type=int, default=4, help='Epoch Size' )
    ps.add_argument( '--learning_ratio', '-l', type=float, default=0.001, help='Learning Ratio' )
    ps.add_argument( '--optimizer', '-o', default='Adam', help='Use Optimizer (Nadam or pytouch Optimizer)' )
    ps.add_argument( '--temporary_file', '-t', default='temporary_cv.pickle', help='Temporary File' )
    ps.add_argument( '--score_output', '-s', default='score.txt', help='Output File' )
    ps.add_argument( '--vervose', '-v', action='store_true', help='Vervose' )
    ps.add_argument( '--gpu', '-g', type=int, default=0, help='Use Gpu Device No.' )
    ps.add_argument( '--random_seed', '-r', type=int, default=1234, help='Random Seed' )
    args = ps.parse_args()
    cuda = torch.device('cuda:%d'%args.gpu)
    torch.cuda.set_device(cuda)
    seed_everything()
    if not os.path.isfile(args.temporary_file):
        make_temp()
    v = get_score()
    if args.minimize < 2:
        nepoch = np.max(list(map(int,v.keys())))+1
        epochensemble_scores = []
        singlemodel_scores = []
        for e in range(nepoch):
            nfold = len(v[e])
            epochensemble_score = 0
            singlemodel_score = 0
            for f in range(nfold):
                epochensemble_score += v[e][f][0]
                singlemodel_score += v[e][f][1]
            epochensemble_score = epochensemble_score / nfold
            singlemodel_score = singlemodel_score / nfold
            epochensemble_scores.append(epochensemble_score)
            singlemodel_scores.append(singlemodel_score)
        print('Scores:')
        print('epoch\te-ensemble\tsingle')
        for e in range(nepoch):
            print('%d\t%f\t%f'%(e+1,epochensemble_scores[e],singlemodel_scores[e]))
        print('BestScore : ',max(np.max(epochensemble_scores),np.max(singlemodel_scores)),': in %s epoch %d'%( ('single' if np.max(singlemodel_scores) > np.max(epochensemble_scores) else 'ensemble'), (np.argmax(singlemodel_scores)+1 if np.max(singlemodel_scores) > np.max(epochensemble_scores) else np.argmax(epochensemble_scores)+1) ))
        with open(args.score_output,'w') as f:
            for k,i in vars(args).items():
                f.write(str(k))
                f.write(' : ')
                f.write(str(i))
                f.write('\n')
            f.write('\n')
            f.write('Scores:\n')
            f.write('epoch\te-ensemble\tsingle\n')
            for e in range(nepoch):
                f.write('%d\t%f\t%f\n'%(e+1,epochensemble_scores[e],singlemodel_scores[e]))
            f.write('\n')
            f.write('BestScore : \n')
            f.write(str(max(np.max(epochensemble_scores),np.max(singlemodel_scores))))
            f.write(': in %s epoch %d\n'%( ('single' if np.max(singlemodel_scores) > np.max(epochensemble_scores) else 'ensemble'), (np.argmax(singlemodel_scores)+1 if np.max(singlemodel_scores) > np.max(epochensemble_scores) else np.argmax(epochensemble_scores)+1) ))
            f.write('BestSingleScore : \n')
            f.write(str(np.max(singlemodel_scores)))
            f.write(': in epoch %d'%( np.argmax(singlemodel_scores)+1 ))
    else:
        print('SingleScore : %f'%v)
        with open(args.score_output,'w') as f:
            for k,i in vars(args).items():
                f.write(str(k))
                f.write(' : ')
                f.write(str(i))
                f.write('\n')
            f.write('\n')
            f.write('SingleScore : \n')
            f.write(str(v))
