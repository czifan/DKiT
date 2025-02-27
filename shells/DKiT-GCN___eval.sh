#!/bin/bash

model_type="Ours"
model_name="DKiT-GNN"
num_states=3
num_stations=16
embedding_dim=64
noise_std=0.0
num_encoder_layers=4
num_decoder_layers=2
hidden_dim=256
num_heads=1
num_classes=2
device="cuda:0"
chk_file="./results_20250226/Ours/DKiT-GNN/K1+K2+K3/1740566526.1153634/last.pth"
gnn_adjmatrix="K1+K2+K3"

gnn_name="GCN"
gnn_in_channels=64
gnn_mid_channels=64
gnn_out_channels=64

dataset_name="DKiTDataset"
data_path="./Data/mask_tasks/"
train_file="train.csv"
eval_file="test.csv"
eval_patient_ids="./Data/mask_tasks/test_ids.txt"
raw_data_file="./Data/raw_data.csv"
num_workers=1

seed=0
criterion_name="DKiTCriterion"
ignore_index=-1

system_N_init=3
system_recommend_K=3
system_N_sampling=5

output_path=$(dirname "$chk_file")

for system_N_init in {1,3,5}
do
python eval.py \
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
  --chk_file $chk_file \
  --gnn_name $gnn_name \
  --gnn_adjmatrix $gnn_adjmatrix \
  --gnn_in_channels $gnn_in_channels \
  --gnn_mid_channels $gnn_mid_channels \
  --gnn_out_channels $gnn_out_channels \
  --dataset_name $dataset_name \
  --data_path $data_path \
  --train_file $train_file \
  --eval_file $eval_file \
  --eval_patient_ids $eval_patient_ids \
  --raw_data_file $raw_data_file \
  --num_workers $num_workers \
  --seed $seed \
  --criterion_name $criterion_name \
  --ignore_index $ignore_index \
  --system_N_init $system_N_init \
  --system_recommend_K $system_recommend_K \
  --system_N_sampling $system_N_sampling \
  --output_path $output_path \
  --system_N_init $system_N_init
done 