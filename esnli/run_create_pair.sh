split=$1
python3 create_esnli_s2s_pair.py \
  -sentA preprocessed/${split}/sentenceA.txt \
  -sentB preprocessed/${split}/sentenceB.txt \
  -label preprocessed/${split}/label.txt \
  -highlightA preprocessed/${split}/highlightA.txt \
  -highlightB preprocessed/${split}/highlightB.txt \
  -out preprocessed/${split}/esnli_sents_highlight_contradict_pairs.tsv \
  -class 'contradiction' \
  -target highlight

python3 create_esnli_s2s_pair.py \
  -sentA preprocessed/${split}/sentenceA.txt \
  -sentB preprocessed/${split}/sentenceB.txt \
  -label preprocessed/${split}/label.txt \
  -highlightA preprocessed/${split}/highlightA.txt \
  -highlightB preprocessed/${split}/highlightB.txt \
  -out preprocessed/${split}/esnli_sents_highlight_all_pairs.tsv \
  -target highlight

python3 create_esnli_s2s_pair.py \
  -sentA preprocessed/${split}/sentenceA.txt \
  -sentB preprocessed/${split}/sentenceB.txt \
  -label preprocessed/${split}/label.txt \
  -highlightA preprocessed/${split}/highlightA.txt \
  -highlightB preprocessed/${split}/highlightB.txt \
  -out preprocessed/${split}/esnli_sents_highlight_ctrl_pairs.tsv \
  -class 'all' \
  -target 'highlight_ctrl'
~                             
