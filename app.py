from flask import Flask, request, jsonify
import anthropic
import os
from collections import defaultdict

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Conversation history per user: {user_id: [{"role": ..., "content": ...}]}
conversation_history = defaultdict(list)
MAX_TURNS = 10  # Remember last 10 exchanges

SYSTEM_PROMPT = """أنت المنسقة — مسؤولة خدمة العملاء لكورسات محادثة إنجليزي أونلاين على إنستغرام.

══════════════════════════════════
قواعد أساسية
══════════════════════════════════
- دايماً رد بالعربي الأردني — ما تستخدم إنجليزي إلا لأسماء الكورسات والمصطلحات التقنية
- رسائل قصيرة ومباشرة — ما حدا بيحب رسائل طويلة بالـ DM
- دايماً اختم بسؤال أو جملة تفتح باب الحوار — ما تنهي المحادثة أبداً
- استخدم إيموجي باعتدال (2-3 بالرسالة)
- عدّل الضمائر حسب جنس الشخص إذا واضح (جاهز/جاهزة، احكيلنا/احكيلينا)
- ما تقول إنك بوت أو ذكاء اصطناعي — أنت "المنسقة"
- إذا ما بتعرف الجواب قول: "رح نرجعلكم بوقت قريب 🙂" — ما تخترع معلومات أبداً
- ما تتكلم عن أي موضوع مش متعلق بالكورسات"""


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get('user_id')
    message = data.get('message', '').strip()

    if not user_id or not message:
        return jsonify({"error": "Missing user_id or message"}), 400

    conversation_history[user_id].append({"role": "user", "content": message})

    if len(conversation_history[user_id]) > MAX_TURNS * 2:
        conversation_history[user_id] = conversation_history[user_id][-(MAX_TURNS * 2):]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=conversation_history[user_id]
    )

    reply = response.content[0].text
    conversation_history[user_id].append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
