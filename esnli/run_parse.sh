# train 
# python3 parse_esnli.py \
#     -train1 esnli_train_1.csv \
#     -train2 esnli_train_2.csv \
#     -dev esnli_dev.csv \
#     -test esnli_test.csv \
#     -split train \
#     -output_dir preprocessed

# dev 
python3 parse_esnli.py \
    -train1 esnli_train_1.csv \
    -train2 esnli_train_2.csv \
    -dev esnli_dev.csv \
    -test esnli_test.csv \
    -split dev \
    -output_dir preprocessed

# test 
# python3 parse_esnli.py \
#     -train1 esnli_train_1.csv \
#     -train2 esnli_train_2.csv \
#     -dev esnli_dev.csv \
#     -test esnli_test.csv \
#     -split test \
#     -output_dir preprocessed
