from llama_cpp import Llama
from pathlib import Path
import json

model_path = Path(__file__).parent / "models/TinySwallow-1.5B-Instruct-Q4_K_L.gguf"

system_prompt = """
以下の注文文から以下項目を抽出し、JSON形式で出力
・品目
・数量
・サイズ

フォーマットは
{
  "orders": [
    {
        "item_name": 品目,
        "quantity": 数量,
        "size": サイズ,
    },
  ]
}

・item_nameからサイズ指定の単語は除去。例えば「牛丼大盛り」→ item_name: "牛丼", size: "大"。例えば「小カレー」→ item_name: "カレー", size: "小"
・size: "大","中","小"のいずれかで出力
・size: 以下ルールで正規化
1. "大盛り", "多め","L","ラージサイズ"→ "大"
2. "普通盛り","M","ミディアムサイズ"→ "中"
3. "小盛り","少なめ","S","スモールサイズ"→ "小"
・size: 指定がない場合は"中"
"""

def analyze_order(system_prompt, user_prompt):
    # モデル読み込み
    model = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        verbose=False,
        n_threads=4,
    )

    # LLM予測
    messages = [
        {
            "role": "system",
            "content":  system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    response = model.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=128,
    )

    response_str = response["choices"][0]["message"]["content"]

    # JSONオブジェクトに変換
    try:
        json_data = json.loads(response_str)
    except ValueError:
        raise ValueError(f"Invalid json format: {response_str}")

    return json_data

if __name__ == "__main__":
    # ユーザーの注文文
    user_prompt = "牛丼とカレーを３つずつ"
    #user_prompt = "牛丼大盛り１つとカレー小を２つください"
    #user_prompt = "牛丼特盛り１つとカレー小を２つください"
    #user_prompt = "牛丼ラージサイズ１つとカレー小を２つください"
    #user_prompt = "牛丼L１つとカレー小を２つください"
    #user_prompt = "牛丼Lサイズ１つとカレー小を２つください"
    #user_prompt = "牛丼大盛り１つとカレースモールサイズを２つください"
    #user_prompt = "牛丼多め１つと少なめのカレーを２つください"
    #user_prompt = "牛丼多め１つとカレー少なめを２つください"
    #user_prompt = "牛丼の小さいの１つと小カレーを２つください"

    # ユーザーの注文文から品目と数量を抽出してJSON形式で取得
    json_data = analyze_order(system_prompt, user_prompt)

    # 結果表示（見やすくインデントを付けて表示）
    print(user_prompt)
    print(json.dumps(json_data, indent=2, ensure_ascii=False))
