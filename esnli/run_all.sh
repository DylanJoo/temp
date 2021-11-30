bash run_parser.shZZ

bash run_create_pairs.sh "train"
bash run_create_pairs.sh "dev"
bash run_create_pairs.sh "test"

bash run_create_highlight_list.sh "train"
bash run_create_highlight_list.sh "dev"
bash run_create_highlight_list.sh "test"
