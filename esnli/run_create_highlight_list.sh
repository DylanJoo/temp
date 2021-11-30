split=$1
python3 create_esnli_token_clf.py \
    -sentA preprocessed/${split}/sentenceA.txt \
    -sentB preprocessed/${split}/sentenceB.txt \
    -highlightA preprocessed/${split}/highlightA.txt \
    -highlightB preprocessed/${split}/highlightB.txt \
    -label preprocessed/${split}/label.txt \
    -out preprocessed/${split}/esnli_sents_highlight_all.jsonl \
    -class 'all'

python3 create_esnli_token_clf.py \
    -sentA preprocessed/${split}/sentenceA.txt \
    -sentB preprocessed/${split}/sentenceB.txt \
    -highlightA preprocessed/${split}/highlightA.txt \
    -highlightB preprocessed/${split}/highlightB.txt \
    -label preprocessed/${split}/label.txt \
    -out preprocessed/${split}/esnli_sents_highlight_contradict.jsonl \
    -class 'contradiction'
