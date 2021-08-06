echo "Downloading QuAC dataset: ..."
wget -q https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O train_quac.json
wget -q https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O val_quac.json

echo "Downloading canard datasett: ..."
wget -q https://github.com/aagohary/canard/raw/master/data/release/train.json -O train_canard.json

