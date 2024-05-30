import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import streamlit as st

# ローカルの GPT-Neo モデルとトークナイザーを読み込む
model_path = "EleutherAI/gpt-neo-1.3B"
tokenizer_path = "EleutherAI/gpt-neo-1.3B"

# トークナイザーとモデルをインスタンス化
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)

# タイトルを設定
st.title("ChatGPT by Streamlit")

# セッション内のメッセージが指定されていない場合のデフォルト値
if "messages" not in st.session_state:
    st.session_state.messages = []

# 以前のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの新しい入力を取得
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 一時的なプレースホルダーを作成
        full_response = ""
        # モデルからの応答を生成
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response}) # 応答をメッセージに追加

