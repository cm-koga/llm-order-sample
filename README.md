# llm-order-sample

Inferface誌 LLM記事のサンプルコード

# 環境構築
* Python 3.10以上

## Pythonパッケージのインストール
以下コマンドを実行するとpythonパッケージをインストールします。
```
pip install -r requirements.txt
```

## モデルダウンロード
以下コマンドを実行するとmodelsディレクトリ下にモデルファイルがダウンロードされます。
※モデルファイルは1.04Gほどのサイズとなります。

```
python download_model.py
```
モデルは以下のモデルを使っています。
https://huggingface.co/bartowski/TinySwallow-1.5B-Instruct-GGUF

# サンプルコードの実行
サンプルコードの詳細についてはInferface誌の記事を参照ください

```
python sample_001.py
```
