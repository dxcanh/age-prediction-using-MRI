#! /bin/bash
model=ScaleDense
batch_size=16
test_dirpath=/home/canhdx/workspace/TSAN-brain-age-estimation/data/test
excel_dirpath=/home/canhdx/workspace/TSAN-brain-age-estimation/label.xlsx
sorter_path=/home/canhdx/workspace/TSAN-brain-age-estimation/TSAN/Sodeep_pretrain_weight/Tied_rank_best_lstmla_slen_16.pth.tar
model_dirpath=/home/canhdx/workspace/TSAN-brain-age-estimation/pretrained_model/second_stage_test/ScaleDense_best_model.pth.tar
first_stage_net=/home/canhdx/workspace/TSAN-brain-age-estimation/pretrained_model/ScaleDenseScaleDense_best_model.pth.tar

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=1 python /home/canhdx/workspace/TSAN-brain-age-estimation/TSAN/prediction_second_stage.py \
--model             ${model}                             \
--batch_size        $batch_size                          \
--output_dir        ${model_dirpath}                     \
--model_name        ${model}          \
--test_folder       ${test_dirpath}                      \
--excel_path        ${excel_dirpath}                     \
--npz_name          /home/canhdx/workspace/TSAN-brain-age-estimation/brain_age.npz                      \
--dis_range         5                                    \
--first_stage_net   ${first_stage_net}                   \
--sorter            ${sorter_path}                       \

# ============= Hyperparameter Description ============== #
# --model             Deep learning model to do brain age estimation
# --batch_size        Batch size during training process
# --output_dir        Output dictionary, whici will contains training log and model checkpoint
# --model_name        Checkpoint file name
# --test_folder       Test set data path
# --excel_path        Excel file path
# --npz_name          npz file name to store predited brain age
# --dis_range         Discritize step when training the second stage network
# --first_stage_net   When training the second stage network, appoint the trained first stage network checkpoint file path is needed
# --sorter            When use ranking loss, the pretrained SoDeep sorter network weight need to be appointed


