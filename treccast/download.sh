#echo "Downloading QuAC_eqa dataset: ..."
#wget -q https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O train_quac.json
#wget -q https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O val_quac.json
#
#echo "Downloading Quac_canard datasett: ..."
#wget -q https://github.com/aagohary/canard/raw/master/data/release/train.json -O train_canard.json

echo "Downloading Quac_lif datasett: ..."
wget "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun" -O lif_v1.zip && rm -rf /tmp/cookies.txt
unzip lif_v1.zip
rm lif_v1.zip
