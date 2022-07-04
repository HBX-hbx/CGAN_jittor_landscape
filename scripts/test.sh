cd ..
name=test_spade0627_debug_test1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CUDA_VISIBLE_DEVICES=1 \
python3.7 test.py \
--name ${name} \
--output_path ./results/${name} \
--checkpoints_dir ./results \
--results_dir ./test_results \
--data_path ./data \
--batch_size 1 \
--label_nc 29 \
--no_instance \
--preprocess_mode "scale_width" \
--aspect_ratio 1.3333 \
