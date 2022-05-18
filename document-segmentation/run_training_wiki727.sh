python3 train.py \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --instance_per_example 2 \
  --remove_unused_columns False \
  --train_folder /wiki_727/train \
  --eval_folder /wiki_727/dev \
  --test_folder /wiki_727/test \
  --max_seq_length 512
