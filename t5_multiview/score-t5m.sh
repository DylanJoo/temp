for ITER in {00..20}; do
  echo "Running iter: $ITER" >> process-t5m.out
  nohup t5_mesh_transformer \
    --tpu="tpu-pr3" \
    --gcp_project="trec-cast-322304" \
    --tpu_zone="europe-west4-a" \
    --model_dir="gs://conversational-ir/pr-pointwise_ranking/models/large-multiview" \
    --gin_file="gs://t5-data/pretrained_models/large/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="score_from_file.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="infer_checkpoint_step = 1100700" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
    --gin_param="Bitransformer.decode.max_decode_length = 2" \
    --gin_param="inputs_filename = 'gs://conversational-ir/treccast/data/passage_reranking/autocano_official/queries_autocano_text_pair.tsv$ITER'" \
    --gin_param="targets_filename = 'gs://conversational-ir/pr-pointwise_ranking/data/true_token.txt'" \
    --gin_param="scores_filename = 'gs://conversational-ir/treccast/data/passage_reranking/autocano_official/scores/t5m_query_passage_relevance.txt$ITER'" \
    --gin_param="Bitransformer.decode.beam_size = 1" \
    --gin_param="Bitransformer.decode.temperature = 0.0" \
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
    >> split-t5m.out 2>&1
done &
echo $! >> process-t5m.out 

