# phonikud-byt5

Hebrew G2P with byt5 model based on [Phonikud](https://phonikud.github.io)

<img width="800" alt="image" src="https://github.com/user-attachments/assets/67e57b8c-372e-4dac-bd60-7db8a5ebe705" />

Checkpoints available at https://huggingface.co/thewh1teagle/phonikud-byt5


## Prepare

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
sudo apt install -y p7zip-full wget
wget https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
7z x knesset_phonemes_v1.txt.7z
mkdir data
head -n 2000000 knesset_phonemes_v1.txt > ./data/data.txt
```

Unvocalized data.txt

```console
uv run python -c "import re, pathlib; p=pathlib.Path('data/data.txt'); txt=p.read_text(encoding='utf-8'); txt=re.sub(r'[|\u0590-\u05cf]', '', txt); pathlib.Path('data/data_clean.txt').write_text(txt, encoding='utf-8')"
rm -rf ./data/data.txt
head ./data/data_clean.txt
```

```console
uv run src/phonikud_byt5/run_train.py
```

## Logs

```console
uv run wandb login
```

## Onnx

See [byt5-onnx](./byt5-onnx/)

## Monitor training in Telegram

```console
CHAT_ID="-1002594452051"
BOT_TOKEN="bot token"

while true; do
  # run prediction
  OUTPUT=$(uv run src/phonikud_byt5/run_predict.py)

  # extract best_model step and loss from metadata.json
  STEP=$(jq -r '.best_model.step' ./checkpoints/metadata.json)
  LOSS=$(jq -r '.best_model.eval_loss' ./checkpoints/metadata.json)

  # save to file
  echo "$OUTPUT" > /tmp/g2p_result.txt

  # send to telegram
  curl -s "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${CHAT_ID}" \
    --data-urlencode "text=G2P result v1 (step: $STEP, loss: $LOSS):\n$OUTPUT"

  # wait 20 min
  sleep 1200
done
```
