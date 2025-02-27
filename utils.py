import random
import logging
import numpy as np 
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from models import *
from dataset import *
from criterion import *
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, roc_curve 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def build_logger(log_file):
    logger = logging.getLogger('DKiT')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def build_model(args):
    if args.model_type == 'Ours':
        if args.model_name == 'DKiT':
            model = DKiT(
                num_states=args.num_states,
                num_stations=args.num_stations,
                embedding_dim=args.embedding_dim,
                noise_std=args.noise_std,
                use_noise=True,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                num_classes=args.num_classes
            )
        elif args.model_name == 'DKiT-GNN':
            model = DKiT_GNN(
                num_states=args.num_states,
                num_stations=args.num_stations,
                embedding_dim=args.embedding_dim,
                noise_std=args.noise_std,
                use_noise=True,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                num_classes=args.num_classes,
                gnn_name=args.gnn_name,
                gnn_adjmatrix=args.gnn_adjmatrix,
                gnn_in_channels=args.gnn_in_channels,
                gnn_mid_channels=args.gnn_mid_channels,
                gnn_out_channels=args.gnn_out_channels
            )
        else:
            raise NotImplementedError(f"Model name {args.model_name} not implemented")
    elif args.model_type == 'Stats':
        # if args.model_name == "StatsMode":
        #     model = StatsMode()
        if "Percentile" in args.model_name:
            model = StatsPercentile(p=args.stats_percentile)
        else:
            model = eval(args.model_name)()
    elif args.model_type == "ML":
        if args.model_name == "LR":
            model = LogisticRegression(C=0.1, solver='liblinear')
        elif args.model_name == "RF":
            model = RandomForestClassifier(n_estimators=20, max_depth=10)
        elif args.model_name == "SVM":
            model = SVC(kernel='linear', C=0.1, probability=True, random_state=args.seed)
        elif args.model_name == "DT":
            model = DecisionTreeClassifier(max_depth=8, random_state=args.seed)
    elif args.model_type == "Graph":
        model = GraphModel(
            num_states=args.num_states,
            num_stations=args.num_stations,
            embedding_dim=args.embedding_dim,
            noise_std=args.noise_std,
            use_noise=True,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_classes=args.num_classes,
            gnn_name=args.gnn_name,
            gnn_adjmatrix=args.gnn_adjmatrix,
            gnn_in_channels=args.gnn_in_channels,
            gnn_mid_channels=args.gnn_mid_channels,
            gnn_out_channels=args.gnn_out_channels
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    return model

def build_criterion(args):
    if args.criterion_name == "CE":
        return nn.CrossEntropyLoss()
    elif args.criterion_name == "MSE":
        return nn.MSELoss()
    elif args.criterion_name == "DKiTCriterion":
        return DKiTCriterion(ignore_index=args.ignore_index)
    elif args.criterion_name == "IdentityCriterion":
        return IdentityCriterion()
    else:
        raise NotImplementedError(f"Criterion name {args.criterion_name} not implemented")
    
def build_optimizer(model, args):
    if args.optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer name {args.optimizer_name} not implemented")
    return optimizer

def build_scheduler(optimizer, args, printer=print):
    if args.scheduler_name == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler_name == 'cosine_lr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    elif args.scheduler_name == 'exp_lr':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    else:
        printer(f"Scheduler {args.scheduler_name} not implemented")
        scheduler = None
    return scheduler

def build_dataloader(args, split_file, shuffle=True, printer=print):
    if args.dataset_name == "DKiTDataset":
        dataset = DKiTDataset(split_file, printer=printer)
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=shuffle)
    elif args.dataset_name == "MLDataset":
        dataset = MLDataset(split_file, printer=printer)
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=shuffle)
    else:
        raise NotImplementedError(f"Dataset name {args.dataset_name} not implemented")
    return dataset, dataloader

def plot(epoch_lst, train_loss_lst, train_auc_lst, valid_loss_lst, valid_auc_lst, save_file):
    _, ax = plt.subplots(1, 2, figsize=(27, 5))
    ax[0].plot(epoch_lst, train_loss_lst, label='Train Loss')
    ax[0].scatter(epoch_lst, train_loss_lst, s=16, marker='o')
    ax[0].plot(epoch_lst, valid_loss_lst, label='Valid Loss', linestyle='--')
    ax[0].scatter(epoch_lst, valid_loss_lst, s=16, marker='^')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epoch_lst, train_auc_lst, label='Train AUC')
    ax[1].scatter(epoch_lst, train_auc_lst, s=16, marker='o')
    ax[1].plot(epoch_lst, valid_auc_lst, label='Valid AUC', linestyle='--')
    ax[1].scatter(epoch_lst, valid_auc_lst, s=16, marker='^')
    ax[1].legend()
    ax[1].grid(True)

    plt.savefig(save_file)
    plt.close()

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.clone().detach()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name] = self.shadow[name] * self.decay + param.data * (1.0 - self.decay)

    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow[name])

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.shadow[name])


def AUROC_LN(y, pred):
    y = np.array(y)
    pred = np.array(pred)
    keep_idx = y != -1
    pred = pred[keep_idx]
    y = y[keep_idx]
    return roc_auc_score(y, pred)

def AUROC_LN_ci(total_labels, total_preds, n_bootstraps=2000, ci=95):
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)
    keep_idx = total_labels != -1
    total_preds = total_preds[keep_idx]
    total_labels = total_labels[keep_idx]

    auc = AUROC_LN(total_labels, total_preds)
    bootstrapped_aucs = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(total_labels), len(total_labels))
        if len(np.unique(total_labels[indices])) < 2:
            continue
        auc_bootstrap = AUROC_LN(total_labels[indices], total_preds[indices])
        bootstrapped_aucs.append(auc_bootstrap)
    lower_bound = np.percentile(bootstrapped_aucs, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_aucs, 100 - (100 - ci) / 2)
    return auc, lower_bound, upper_bound

def compute_p_at_r(true_labels, predicted_scores, recall_thresholds=[0.75, 0.85, 0.95]):
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)
    keep_idx = true_labels != -1
    predicted_scores = predicted_scores[keep_idx]
    true_labels = true_labels[keep_idx]

    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_labels = true_labels[sorted_indices]

    total_relevant_items = np.sum(true_labels)
    
    retrieved_relevant = 0
    results = {}
    for k in range(1, len(true_labels) + 1):
        retrieved_relevant += sorted_labels[k - 1]
        recall = retrieved_relevant / total_relevant_items
        
        if recall >= recall_thresholds[0]:
            precision = retrieved_relevant / k
            results[f"P@R{int(recall_thresholds[0]*100)}"] = precision
            recall_thresholds.pop(0)
            if len(recall_thresholds) == 0:
                break
                
    return results

def compute_f1_score(true_labels, predicted_scores):
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)
    keep_idx = true_labels != -1
    predicted_scores = predicted_scores[keep_idx]
    true_labels = true_labels[keep_idx]

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    youden_idx = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_idx)]
    
    predicted_labels = (predicted_scores > best_threshold).astype(int)
    return f1_score(true_labels, predicted_labels)

def compute_threshold(true_labels, predicted_scores):
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)
    keep_idx = true_labels != -1
    predicted_scores = predicted_scores[keep_idx]
    true_labels = true_labels[keep_idx]

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
    youden_idx = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_idx)]
    return best_threshold