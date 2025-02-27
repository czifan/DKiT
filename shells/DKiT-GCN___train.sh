#!/bin/bash

model_type="Ours"
model_name="DKiT-GNN"
num_states=3
num_stations=16
embedding_dim=64
noise_std=0.2
num_encoder_layers=4
num_decoder_layers=2
hidden_dim=256
num_heads=1
num_classes=2
device="cuda:0"

gnn_name="GCN"
# gnn_adjmatrix="K1+K2+K3"
gnn_in_channels=64
gnn_mid_channels=64
gnn_out_channels=64

dataset_name="DKiTDataset"
data_path="./Data/mask_tasks/"
train_file="train.csv"
valid_file="valid.csv"
test_file="test.csv"
num_workers=1

seed=0
optimizer_name="AdamW"
lr=1e-3
batch_size=128
epochs=400
weight_decay=1e-4
scheduler_name="cosine_lr"
T_max=100
eta_min=1e-5
criterion_name="DKiTCriterion"
ignore_index=-1
ema_decay=0.999

output_path="./results_20250226/"

for gnn_adjmatrix in "K1" "K2" "K3" "K1+K2+K3"
do 
python train.py \
  --model_type $model_type \
  --model_name $model_name \
  --num_states $num_states \
  --num_stations $num_stations \
  --embedding_dim $embedding_dim \
  --noise_std $noise_std \
  --num_encoder_layers $num_encoder_layers \
  --num_decoder_layers $num_decoder_layers \
  --hidden_dim $hidden_dim \
  --num_heads $num_heads \
  --num_classes $num_classes \
  --device $device \
  --gnn_name $gnn_name \
  --gnn_adjmatrix $gnn_adjmatrix \
  --gnn_in_channels $gnn_in_channels \
  --gnn_mid_channels $gnn_mid_channels \
  --gnn_out_channels $gnn_out_channels \
  --dataset_name $dataset_name \
  --data_path $data_path \
  --train_file $train_file \
  --valid_file $valid_file \
  --test_file $test_file \
  --num_workers $num_workers \
  --seed $seed \
  --optimizer_name $optimizer_name \
  --lr $lr \
  --batch_size $batch_size \
  --epochs $epochs \
  --weight_decay $weight_decay \
  --scheduler_name $scheduler_name \
  --T_max $T_max \
  --eta_min $eta_min \
  --criterion_name $criterion_name \
  --ignore_index $ignore_index \
  --ema_decay $ema_decay \
  --output_path $output_path
done