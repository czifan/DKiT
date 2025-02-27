import argparse
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

from utils import *
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train DKiT model')
    # Model arguments
    parser.add_argument('--model_type', type=str, default='Ours', choices=['Ours', 'Stats', 'ML', 'Graph'])
    parser.add_argument('--model_name', type=str, default='DKiT', choices=['DKiT', 'DKiT-GNN', 'StatsMode', 'StatsMean', 'StatsPercentile', 'LR', 'SVM', 'RF', 'DT', 'GCN', 'GIN', 'GAT'])
    parser.add_argument('--num_states', type=int, default=3)
    parser.add_argument('--num_stations', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--noise_std', type=float, default=0.2)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stats_percentile', type=int, default=50)

    parser.add_argument('--gnn_name', type=str, default='GCN', choices=['GCN', 'GIN', 'GAT'])
    parser.add_argument('--gnn_adjmatrix', type=str, default='K1', choices=['K1', 'K2', 'K3', 'K1+K2+K3'])
    parser.add_argument('--gnn_in_channels', type=int, default=64)
    parser.add_argument('--gnn_mid_channels', type=int, default=64)
    parser.add_argument('--gnn_out_channels', type=int, default=64)

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='DKiTDataset')
    parser.add_argument('--data_path', type=str, default='./Data/mask_tasks/')
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--valid_file', type=str, default='valid.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--eval_file', type=str, default='eval.csv')
    parser.add_argument('--eval_patient_ids', type=str, default='eval_patient_ids.csv')
    parser.add_argument('--raw_data_file', type=str, default='raw_data.csv')
    parser.add_argument('--num_workers', type=int, default=1)

    # Optimization arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer_name', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_name', type=str, default='cosine_lr', choices=['step_lr', 'cosine_lr', 'exp_lr'])
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--T_max', type=int, default=100)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--criterion_name', type=str, default='DKiTCriterion', choices=['DKiTCriterion', 'CE', 'MSE', 'IdentityCriterion'])
    parser.add_argument('--ignore_index', type=int, default=-1)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # Logging arguments
    parser.add_argument('--output_path', type=str, default='./results/')
    parser.add_argument('--chk_file', type=str, default='None')

    parser.add_argument('--system_N_init', type=int, default=3)
    parser.add_argument('--system_recommend_K', type=int, default=3)
    parser.add_argument('--system_N_sampling', type=int, default=10)

    return parser.parse_args()

def train_one_epoch(model, ema_model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = []
    total_preds = []
    total_labels = []
    total_aucs = []

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}', ncols=100, leave=True)
    for batch_idx, (X, y, M) in enumerate(progress_bar):
        X, y, M = X.to(device), y.to(device), M.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        ema_model.update()

        total_loss.append(loss.item())

        flatten_pred = torch.softmax(pred, dim=-1)[..., 1].view(-1)
        flatten_y = y.view(-1)
        total_preds.extend(flatten_pred.detach().cpu().numpy())
        total_labels.extend(flatten_y.cpu().numpy())

        auc = AUROC_LN(total_labels, total_preds)
        total_aucs.append(auc)

        progress_bar.set_postfix(loss=f'{loss.item():.4f} ({np.mean(total_loss):.4f})', 
                                 auc=f'{auc:.3f} ({np.mean(total_aucs):.3f})')

    return np.mean(total_loss), np.mean(total_aucs)

def eval_one_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval() 
    total_loss = []
    total_preds = []
    total_labels = []
    total_aucs = []

    progress_bar = tqdm(val_loader, desc=f'> Epoch {epoch+1}/{total_epochs}', ncols=100, leave=True)
    with torch.no_grad():  
        for batch_idx, (X, y, M) in enumerate(progress_bar):
            X, y, M = X.to(device), y.to(device), M.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            total_loss.append(loss.item())

            flatten_pred = torch.softmax(pred, dim=-1)[..., 1].view(-1)
            flatten_y = y.view(-1)
            total_preds.extend(flatten_pred.cpu().numpy())
            total_labels.extend(flatten_y.cpu().numpy())

            auc = AUROC_LN(total_labels, total_preds)
            total_aucs.append(auc)

            progress_bar.set_postfix(loss=f'{loss.item():.4f} ({np.mean(total_loss):.4f})', 
                                     auc=f'{auc:.3f} ({np.mean(total_aucs):.3f})')

    return np.mean(total_loss), np.mean(total_aucs)

def run_ml_models(args, printer):
    train_ds = eval(args.dataset_name)(os.path.join(args.data_path, args.train_file))
    train_X, train_y = train_ds.X, train_ds.y
    valid_ds = eval(args.dataset_name)(os.path.join(args.data_path, args.valid_file))
    valid_X, valid_y = valid_ds.X, valid_ds.y

    model = build_model(args)
    model.fit(train_X, train_y)
    joblib.dump(model, os.path.join(args.output_path, f'{args.model_name}.pkl'))

    train_p = model.predict_proba(train_X)[:, 1]
    valid_p = model.predict_proba(valid_X)[:, 1]

    train_auc = roc_auc_score(train_y, train_p)
    valid_auc = roc_auc_score(valid_y, valid_p)

    printer(f"Train AUC: {train_auc:.4f}, Valid AUC: {valid_auc:.4f}")

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.model_name in ["DKiT-GNN", "Graph"]:
        args.output_path = os.path.join(args.output_path, args.model_type, args.model_name, args.gnn_adjmatrix, str(time.time()))
    else:
        args.output_path = os.path.join(args.output_path, args.model_type, args.model_name, str(time.time()))
    os.makedirs(args.output_path, exist_ok=True)
    logger = build_logger(os.path.join(args.output_path, 'train.log'))
    printer = logger.info

    if args.model_type == 'ML':
        run_ml_models(args, printer)
        return

    train_ds, train_dl = build_dataloader(args, os.path.join(args.data_path, args.train_file), shuffle=True, printer=printer)
    valid_ds, valid_dl = build_dataloader(args, os.path.join(args.data_path, args.valid_file), shuffle=False, printer=printer)

    model = build_model(args).to(args.device)
    ema_model = EMA(model, args.ema_decay)
    criterion = build_criterion(args).to(args.device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, printer)
    
    best_valid_auc, best_valid_epoch = 0.0, None
    train_loss_lst, train_auc_lst = [], []
    valid_loss_lst, valid_auc_lst = [], []
    for epoch in range(args.epochs):
        train_loss, train_auc = train_one_epoch(model, ema_model, train_dl, criterion, optimizer, args.device, epoch, args.epochs)
        train_loss_lst.append(train_loss)
        train_auc_lst.append(train_auc)

        ema_model.apply_shadow()
        valid_loss, valid_auc = eval_one_epoch(model, valid_dl, criterion, args.device, epoch, args.epochs)
        valid_loss_lst.append(valid_loss)
        valid_auc_lst.append(valid_auc)

        plot(range(epoch+1), train_loss_lst, train_auc_lst, valid_loss_lst, valid_auc_lst, os.path.join(args.output_path, 'train.png'))

        torch.save(model.state_dict(), os.path.join(args.output_path, 'last.pth'))
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_epoch = epoch 
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best.pth'))

        printer(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {valid_loss:.4f}, Val AUC: {valid_auc:.4f}, Best Epoch: {best_valid_epoch}, Best AUC: {best_valid_auc:.4f}")
        printer('=' * 100)

        if scheduler:
            scheduler.step()

        ema_model.restore()


if __name__ == '__main__':
    main()
