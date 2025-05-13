#! /bin/bash

dis_range=5
model=ScaleDense
loss=mse
batch_size=16
lbd=10
beta=1
first_stage_net=/home/canhdx/workspace/TSAN-brain-age-estimation/pretrained_model_v0/ScaleDenseScaleDense_best_model.pth.tar
save_path=/home/canhdx/workspace/TSAN-brain-age-estimation/pretrained_model/second_stage_test/
label=/home/canhdx/workspace/TSAN-brain-age-estimation/label.xlsx

train_data=/home/canhdx/workspace/TSAN-brain-age-estimation/data/train
valid_data=/home/canhdx/workspace/TSAN-brain-age-estimation/data/val
test_data=/home/canhdx/workspace/TSAN-brain-age-estimation/data/test

sorter_path=/home/canhdx/workspace/TSAN-brain-age-estimation/TSAN/Sodeep_pretrain_weight/Tied_rank_best_lstmla_slen_16.pth.tar

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=2     python /home/canhdx/workspace/TSAN-brain-age-estimation/TSAN/train_second_stage.py       \
--batch_size               $batch_size         \
--epochs                   150                 \
--lr                       1e-5                \
--weight_decay             5e-4                \
--loss                     $loss               \
--aux_loss                 ranking             \
--lbd                      $lbd                \
--beta                     $beta               \
--first_stage_net          ${first_stage_net}  \
--train_folder             ${train_data}       \
--valid_folder             ${valid_data}       \
--test_folder              ${test_data}        \
--excel_path               ${label}            \
--model                    ${model}            \
--output_dir               ${save_path}        \
--dis_range                ${dis_range}        \
--sorter                   ${sorter_path}      \

# ============= Hyperparameter Description ============== #
# --batch_size        Batch size during training process
# --epochs            Total training epochs
# --lr                Initial learning rate
# --weight_decay      L2 weight decay
# --loss              Main loss fuction for training network
# --aux_loss          Auxiliary loss function for training network
# --lbd               The weight between main loss function and auxiliary loss function
# --beta              The weight between ranking loss function and age difference loss function
# --first_stage_net   When training the second stage network, appoint the trained first stage network checkpoint file path is needed
# --train_folder      Train set data path
# --valid_folder      Validation set data path
# --test_folder       Test set data path
# --excel_path        Excel file path
# --model             Deep learning model to do brain age estimation
# --output_dir        Output dictionary, whici will contains training log and model checkpoint
# --dis_range         Discritize step when training the second stage network
# --sorter            When use ranking loss, the pretrained SoDeep sorter network weight need to be appointed