from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import openai
from linebot_ai.rag_search_faiss import semantic_search, build_or_update_faiss_index
import os
import csv
from datetime import datetime, timedelta
import requests

# =====================
# 1. 建立知識庫向量索引
# =====================
build_or_update_faiss_index()

# =====================
# 2. 設定 LINE 和 OpenAI API 金鑰
# =====================
LINE_CHANNEL_ACCESS_TOKEN = 'BrFr7Swn9ctwsjzjICcO1jYFZSWJuCmrxKGz9IO8XqYbmYkO/flGFuGEuM1IqoVxETB7wAUSvMUzaroplwRgTjHlsCgGyQ2MqUD93jHL1AFq4lsDkcN4plzn8VDvptbqTyTLXm04JvO7U9FhbBWvnwdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '91268da2f81eccb934f046c0c7cbc771'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# =====================
# 3. 紀錄查詢 Log 函式
# =====================
LOG_FILE = "chat_log.csv"

def log_interaction(user_id, user_text, reply_text, context_used):
    """紀錄用戶對話與檢索內容"""
    headers = ["timestamp", "user_id", "user_input", "gpt_reply", "retrieved_context"]
    row = [
        datetime.now().isoformat(),
        user_id,
        user_text.replace("\n", " "),
        reply_text.replace("\n", " "),
        context_used.replace("\n", " ")
    ]

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

# =====================
# 4. OpenAI 用量查詢工具
# =====================
def check_openai_api_usage():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 請先設定環境變數 OPENAI_API_KEY")
        return

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    usage_url = "https://api.openai.com/v1/dashboard/billing/usage"
    credits_url = "https://api.openai.com/v1/dashboard/billing/credit_grants"

    today = datetime.utcnow().date()
    start_date = today.replace(day=1).isoformat()
    end_date = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    end_date_str = end_date.isoformat()

    usage_resp = requests.get(
        f"{usage_url}?start_date={start_date}&end_date={end_date_str}",
        headers=headers
    )
    usage = usage_resp.json().get("total_usage", 0) / 100  # 單位：美金

    credits_resp = requests.get(credits_url, headers=headers)
    credits_json = credits_resp.json()
    total_granted = credits_json.get("total_granted", 0)
    total_used = credits_json.get("total_used", 0)
    total_remaining = credits_json.get("total_available", 0)

    print(f"✅ 本月已用：${usage:.2f} USD")
    print(f"✅ 免費額度：總額 ${total_granted:.2f}，已用 ${total_used:.2f}，剩餘 ${total_remaining:.2f}")

# 查一次目前用量
check_openai_api_usage()

# =====================
# 5. LINE Webhook 接收
# =====================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# =====================
# 6. LINE 訊息處理邏輯
# =====================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    user_id = event.source.user_id

    # Step 1：語意檢索
    retrieved_context = semantic_search(user_text, top_k=3)

    # Step 2：送 GPT 回答
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是客服助理，請根據內部知識資料回應用戶問題。"},
            {"role": "user", "content": f"相關知識內容如下：\n{retrieved_context}\n\n使用者問題：{user_text}"}
        ]
    )
    reply_text = response['choices'][0]['message']['content'].strip()

    # Step 3：回 LINE
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

    # Step 4：Log
    log_interaction(user_id, user_text, reply_text, retrieved_context)

# =====================
# 7. 啟動 Flask App
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
