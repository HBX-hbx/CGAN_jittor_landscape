cd ..
name=test_spade0627_debug_test1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CUDA_VISIBLE_DEVICES=1 \
python3.7 train.py \
--name ${name} \
--output_path ./results/${name} \
--checkpoints_dir ./results \
--lr 2e-4 \
--batch_size 1 \
--data_path ./data \
--n_epochs 1 \
--label_nc 29 \
--no_instance \
--preprocess_mode "scale_width" \
--aspect_ratio 1.3333 \
--save_epoch_freq 1 \