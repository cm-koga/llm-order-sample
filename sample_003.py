from llama_cpp import Llama
from pathlib import Path
import json

model_path = Path(__file__).parent / "models/TinySwallow-1.5B-Instruct-Q4_K_L.gguf"

system_prompt = """
以下の注文文から品目と数量を抽出し、JSON形式で出力してください。
JSONだけを出力してください。

フォーマットは
{
  "orders": [
    {
        "item_name": 品目,
        "quantity": 数量,
    },
  ]
}
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
    user_prompt = "牛丼１つとカレーを２つください"

    # ユーザーの注文文から品目と数量を抽出してJSON形式で取得
    json_data = analyze_order(system_prompt, user_prompt)

    # 結果表示（見やすくインデントを付けて表示）
    print(json.dumps(json_data, indent=2, ensure_ascii=False))
