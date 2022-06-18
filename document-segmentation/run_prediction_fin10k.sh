eval_file=$1
python3 inference.py \
  --model_name_or_path 'models/checkpoint-100000' \
  --run_fin10k \
  --pred_dir './prediction' \
  --test_file $eval_file \
  --per_device_eval_batch_size 8 \
  --max_seq_length 512
