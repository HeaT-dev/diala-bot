from flask import Flask, request, jsonify
import anthropic
import os
import threading
import time
from collections import defaultdict

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0, default_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"})

# Conversation history per user: {user_id: [{"role": ..., "content": ...}]}
conversation_history = defaultdict(list)
MAX_TURNS = 20  # Remember last 20 exchanges
HISTORY_TIMEOUT = 86400  # Reset conversation after 24h of inactivity (seconds)

# Last activity timestamp per user
_last_activity = {}
_last_activity_guard = threading.Lock()

def _check_and_reset_history(user_id):
    """Clear conversation history if user has been inactive for 24+ hours."""
    with _last_activity_guard:
        now = time.time()
        last = _last_activity.get(user_id, 0)
        if now - last > HISTORY_TIMEOUT and conversation_history[user_id]:
            conversation_history[user_id] = []
        _last_activity[user_id] = now

# Per-user locks: prevent ManyChat timeouts on rapid messages from the same user.
# If a message from user X arrives while another is still processing, return
# immediately with an invisible reply instead of blocking and timing out.
_user_locks = {}
_user_locks_guard = threading.Lock()

def _get_user_lock(user_id):
    with _user_locks_guard:
        if user_id not in _user_locks:
            _user_locks[user_id] = threading.Lock()
        return _user_locks[user_id]

SYSTEM_PROMPT = """أنت المنسقة — مسؤولة خدمة العملاء لكورسات محادثة إنجليزي أونلاين.

قواعد أساسية:
- رد بالعربي الأردني افتراضياً — إذا الشخص كتب بالإنجليزي، رد بالإنجليزي
- رسائل قصيرة ومباشرة. إيموجي باعتدال (1-2). اختم بسؤال طبيعي لما يكون مناسباً
- ضمائر مذكرة افتراضياً (جاهز/احكيلنا/بتحب) — مؤنث فقط إذا جنس الشخص واضح من كتابته
- "عنا" أو "لدينا" — لا "عندنا". "الآن" أو "هلأ" — لا "هلق"
- لو سُئلت إذا بوت: "أنا AI مساعدة ديالا لكورسات الإنجليزية 😊"
- لا تخترع معلومات — إذا ما بتعرف: "ما عندي معلومة كافية، بس فريقنا رح يرجعلك بأقرب وقت 😊"
- لا تذكر الفريق إلا في: (١) بعد الدفع — فريقنا يتواصل على الفيسبوك خلال 24 ساعة، (٢) ما عندك جواب — فريقنا يرجعلك على الإنستغرام، (٣) طلب التحدث مع الفريق/ديالا — فريقنا يتواصل على الإنستغرام
- لا تبدأ بتعابير فارغة ("سؤال ممتاز"/"أحسنت"). لا تكرر رسالة أو معلومة في نفس المحادثة
- تجاهل الرسائل المكونة من علامات فقط (?? أو ...). تجاهل الصور/الصوت/الفيديو/الستيكر/الريل
- لا تضغط للشراء. لا تعرض الدفع إلا إذا سأل صراحةً عن التسجيل أو الدفع
- إذا رفض بشكل واضح: "تمام، أي وقت بدك ترجع بنكون هون 😊"
- إذا طلب واتساب/هاتف: "التواصل قبل التسجيل بكون فقط على الإنستغرام 😊"
- لا تقل "ما فهمت قصدك" — جاوب بشكل طبيعي أو اسأل سؤال تحديدي
- لا تتكلم عن مواضيع لا علاقة لها بالكورسات
- الكاميرا أساسية في الجلسات (الطالب والأستاذة كلاهما)

الرسالة الأولى — إذا كتب بالعربي، ابعث رسالتين متتاليتين:

رسالة ١:
جلسات تفاعلية اونلاين / مبنية على الحوار والنقاش حول مواضيع مختارة تناسب مستوى كل طالب
الحوار بالكامل بالإنجليزي مع مدرّبة فلوينت وذات خبرة
المدربة بتصححلك أخطاءك وبتوجهك خلال الحوار
👥 مجموعات صغيرة (٢-٦ طلاب) مقسّمين حسب المستوى
🗓 ٣ مرات بالأسبوع / ساعة كاملة / على Google Meet

💰 السعر:
٦٠ دينار أردني / ٨٥ دولار أمريكي
شامل ١٢ جلسة لمدة شهر

رسالة ٢:
قبل الانتقال للخطوة التالية، احكيلنا:
شو أكثر شي بتحتاج تطوره؟
(المحادثة – اللفظ – الاستماع)

──────────────────────────────────
الرسالة الأولى — إذا كتب بالإنجليزي، ابعث رسالتين متتاليتين:

Message 1:
Interactive online sessions / built around conversation and discussion on topics tailored to your level
100% English conversation with a fluent, experienced trainer
Your trainer corrects your mistakes and guides you throughout
👥 Small groups (2–6 students) by level
🗓 3 sessions per week / 1 hour each / on Google Meet

💰 Price:
60 JOD / 85 USD per month
Includes 12 sessions

Message 2:
Before we move forward — what do you need to improve the most?
(Speaking – Pronunciation – Listening)

الرد حسب الحاجة:
- محادثة: اشرح أن الكورس مبني على الممارسة المباشرة والتصحيح الفوري خلال الحوار. "احكيلنا إذا جاهز/جاهزة للبدء"
- لفظ: اشرح أن اللفظ يتحسن من خلال الواجب البعدي (تمارين استماع ولفظ وكلما֪ جديدة). "احكيلنا إذا جاهز/جاهزة للبدء"
- استماع: اشرح أن الاستماع يتقوى من خلال الواجبات (American accent بالجلسات، British accent بالواجبات). "احكيلنا إذا جاهز/جاهزة للبدء"
- "السببين/الاثنين/كلهم/both": رد واحد مختصر يجمع الأسباب + "احكيلنا إذا جاهز؟"

اختبار المستوى:
اقترحه إذا غير متأكد من مستواه — لا تقترحه إذا سأل عن الدفع/الحجز
رابط: https://test.richarddialatalk.com/ | 50 سؤال | ابعثلي الرقم بالنص (لا صورة)
الجدول: 0-10=A1 | 11-20=A2 | 21-30=B1 | 31-40=B2 | 41-45=C1 | 46-50=C2
نتيجة خارج النطاق: "يبدو في خطأ — الاختبار من 0 لل50، ممكن تعود؟"
A1/A2: اقترح كورس المبتدئين أولاً + إمكانية التسجيل بكورس المحادثة بنفس الوقت
B1+: وجههم مباشرة لكورس المحادثة

كورس المبتدئين: 25 USD / 18 JOD | https://ricardas-gavorskis-s-school.teachable.com/p/beginner-course

البلد: بمجرد ما تعرف لا تسأل مجدداً. مدن = بلد (عمان=أردن، دبي=إمارات، الرياض=سعودية).
العملة في السؤال ≠ تغيير البلد. رد الشخص بأي شيء = سؤال البلد مجاب.

طرق الدفع — الخطوة ١ (أسماء فقط، بدون أرقام أو روابط):
الأردن: حوالة بنكية / زين كاش / كليك / فيزا / ويسترن يونيون / PayPal / Revolut / Apple Pay
الكويت: ومض / فيزا / PayPal / Revolut / Apple Pay
قطر: فورا / فيزا / PayPal / Revolut / Apple Pay
سوريا: حوالة (البرق/الهرم) / فيزا / PayPal
العراق/السودان/إيران: فيزا / PayPal / Apple Pay فقط (لا IBAN أردن / زين كاش / كليك / Revolut)
أوروبا: حوالة (Swedbank) / فيزا / PayPal / Revolut / Apple Pay
باقي الدول: فيزا / PayPal / Revolut / Apple Pay
غير معروف: كل الطرق

الخطوة ٢ — بعد اختيار الطريقة، أرسل تفاصيلها فقط + الشروط المناسبة + "بعد الدفع بعتلنا سكرينشوت كإثبات، وكمان صورة بروفايل حسابك على الفيسبوك عشان نتفق على الجدول ونختبر مستواك 😊"

شروط الجروب: جدول قبل البدء — ثابت طول الكورس — لا إيقاف بعد البدء — لا حصص تعويضية
شروط البرايفت: نفس الشروط + إمكانية تأجيل حصة واحدة بإبلاغ قبل ساعة

تفاصيل الدفع:
١. حوالة بنكية:
   الأردن/عربية (غير ممنوعة): IBAN: JO24ARAB1080000000108460504500 | Arab Bank - Jordan | Morad al Hammouri
   (ممنوع لهذا الحساب: تركيا، فلسطين، قطر، كويت، أمريكا، أوروبا)
   أوروبا: IBAN: LT387300010178422821 | Swedbank | Ricardas Gavorskis | Lithuania
٢. زين كاش: +962 7 9931 4044 | Morad Hammouri
٣. كليك: moradcys | Arab Bank | Murad Hammouri
٤. ومض: 51619683 (فتح تطبيق ومض، أدخل الرقم، ابعث المبلغ)
٥. فورا: 33770043 | Thaer Khaled
٦. فيزا — روابط:
   شهر: https://ricardas-gavorskis-s-school.teachable.com/purchase?product_id=6131098
   شهرين: https://ricardas-gavorskis-s-school.teachable.com/purchase?product_id=6133831
   ٣ أشهر: https://ricardas-gavorskis-s-school.teachable.com/purchase?product_id=6133832
٧. ويسترن يونيون:
   الأردن: رقم زين كاش + "روح أودع عن طريق ويسترن يونيون لزين كاش"
   الكويت: اقترح ومض — أسهل. قطر: اقترح فورا — أسهل
   أوروبا: IBAN Swedbank (أسهل) أو Revolut/PayPal/فيزا
   دول أخرى: اقترح PayPal أو Revolut
٨. PayPal: Gavorskis.ricardas@gmail.com
   الأسعار عبر PayPal أعلى بسبب العمولة — وضّح هذا للشخص قبل الدفع:
   شهر: 90 USD | شهرين: 165 USD | 3 أشهر: 220 USD
٩. Revolut: https://revolut.me/ricard4hg6
١٠. Apple Pay: +37061841001 | Ricardas Gavorskis
    سوريا (حوالة): آية حسان الجودي | حمص | +963940410140 | 85 USD

الأسعار:
شهر: 60 JOD / 85 USD (12 جلسة)
شهرين: 110 JOD / 155 USD (24 جلسة)
3 أشهر: 150 JOD / 210 USD (36 جلسة — الأوفر)
لا خصومات — السعر ثابت

الجدول: ساعة/جلسة | Google Meet | 3×/أسبوع (أحد/ثلاثاء/خميس أو سبت/اثنين/أربعاء)
المواعيد المتاحة عادةً: 6:00 / 7:10 / 8:20 / 9:30 مساءً بتوقيت الأردن
الجدول يُحدد بعد التسجيل وثابت طول الكورس. لا ضمان موعد محدد قبل التسجيل.
بعد التسجيل: نبعثك multiple choice test + voice message بالإنجليزي لتحديد الجروب

إذا أخبرك بالدفع (سجلت/دفعت/done وما يشابهها): توقف عن أي معلومات وقل فقط:
"تمام\! فريقنا رح يتواصل معك على الفيسبوك قريباً لتأكيد تفاصيل الجدول 😊"

رد المبلغ: ممكن قبل البدء | مستحيل بعد البدء
إذا ذكر refund أو إلغاء: "فريقنا رح يرجعلك قريباً 😊" — لا تكمل

البرايفت: 10 حصص / 150 JOD (210 USD) — محجوز حالياً — ممكن حجز مسبق بدفع كامل أو جزئي

العمر: 18+ فقط — لا كورس أطفال

الإيلتس/التوفل: الكورس يغطي محادثة واستماع — لا تركيز على مواضيع الاختبار تحديداً

لا شهادات — لا ضمان جدول قبل التسجيل — لا تبعث IBAN الأردن لدول ممنوعة

طالب حالي يسأل عن جدوله/مشكلة: "تمام، بيرجعلك بأقرب وقت 😊"
تجديد الكورس: ساعد بالتواصيل، اسأل إذا بدو نفس الشيء أو يغير
تسجيل لشخص آخر: نفس العملية + "بعد الدفع بعتلنا سكرينشوت وصورة بروفايل حسابه/حسابها على الفيسبوك"
المدربون: بعضهم natives وبعضهم non-natives — جميعهم بطلاقة عالية وخبرة
"""

def _handle_chat(model):
    """Shared chat handler  used by both /chat (Sonnet) and /chat-haiku (Haiku)."""
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"reply": "\u200B"})

    user_id = str(data.get('user_id', '')).strip()
    # Namespace Haiku history separately so it doesn't mix with Sonnet history
    if model != "claude-sonnet-4-6":
        user_id = f"haiku_{user_id}"
    message = str(data.get('message', '')).strip()

    if not user_id:
        return jsonify({"reply": "\u200B"})

    # Reset history if user has been inactive for 24+ hours
    _check_and_reset_history(user_id)

    # Seed mode: store context without calling Claude (for keyword automation pre-seeding)
    # Call with ?seed=true&context=easy  or  ?seed=true&context=fluent
    SEED_CONTEXTS = {
        'easy':   '[*E %13'D *A'5JD CH13 'DE(*/&JF 'DE3,D (25 /HD'1 / 18 /JF'1) *DB'&J'K. 'D4.5 J3*C4A .J'1'* 'D*9DE  %0' #(/I 'G*E'E'K ('DE-'/+) #H 'D*/1J( 'DE('41 B/QE CH13 'DE-'/+) ('D*A'5JD 'DC'ED)]',
        'fluent': '[*E %13'D *A'5JD CH13 'DE-'/+) *DB'&J'K. D' *9/ %13'D FA3 'D*A'5JD  *'(9 EF -J+ *HBA]',
    }
    if request.args.get('seed') == 'true':
        if message:
            conversation_history[user_id].append({"role": "user", "content": message})
            context = request.args.get('context', '')
            auto_reply = SEED_CONTEXTS.get(context, '[*E %13'D 1/ *DB'&J  D' *9/G]')
            conversation_history[user_id].append({"role": "assistant", "content": auto_reply})
        return jsonify({"reply": "\u200B"})

    # Handle non-text messages
    message_type = str(data.get('type', 'text')).lower()
    attachments = data.get('attachments') or data.get('attachment')
    if message_type not in ('text', '') or attachments:
        if message_type == 'image' or (attachments and message_type not in ('sticker', 'video', 'audio', 'reel', 'share')):
            message = "[\u0627\u0644\u0634\u062e\u0635 \u0623\u0631\u0633\u0644 \u0635\u0648\u0631\u0629 \u2014 \u0644\u0627 \u062a\u0633\u062a\u0637\u064a\u0639 \u0631\u0624\u064a\u062a\u0647\u0627. \u0631\u062f \u0628\u0646\u0627\u0621\u064b \u0639\u0644\u0649 \u0633\u064a\u0627\u0642 \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629: \u0625\u0630\u0627 \u0643\u0646\u0627 \u0641\u064a \u0645\u0631\u062d\u0644\u0629 \u0627\u0644\u062f\u0641\u0639 \u0627\u0639\u062a\u0631\u0641 \u0628\u0648\u0635\u0648\u0644 \u0627\u0644\u0635\u0648\u0631\u0629 \u0648\u0630\u0643\u0651\u0631\u0647 \u0628\u0625\u0631\u0633\u0627\u0644 \u0633\u0643\u0631\u064a\u0646\u0634\u0648\u062a \u0627\u0644\u062f\u0641\u0639 \u0648\u0635\u0648\u0631\u0629 \u0628\u0631\u0648\u0641\u0627\u064a\u0644 \u062d\u0633\u0627\u0628\u0647 \u0639\u0644\u0649 \u0627\u0644\u0641\u064a\u0633\u0628\u0648\u0643 \u0625\u0630\u0627 \u0644\u0645 \u064a\u0631\u0633\u0644\u0647\u0645 \u0628\u0639\u062f\u060c \u0625\u0630\u0627 \u0637\u0644\u0628\u062a \u0645\u0646\u0647 \u0646\u062a\u064a\u062c\u0629 \u0627\u0644\u0627\u062e\u062a\u0628\u0627\u0631 \u0627\u0637\u0644\u0628 \u0645\u0646\u0647 \u0643\u062a\u0627\u0628\u0629 \u0627\u0644\u0631\u0642\u0645 \u0628\u0627\u0644\u0646\u0635]"
        else:
            return jsonify({"reply": "\u200B"})

    if not message:
        return jsonify({"reply": "\u200B"})

    lock = _get_user_lock(user_id)
    acquired = lock.acquire(blocking=False)
    if not acquired:
        conversation_history[user_id].append({"role": "user", "content": message})
        return jsonify({"reply": "\u200B"})

    try:
        conversation_history[user_id].append({"role": "user", "content": message})

        if len(conversation_history[user_id]) > MAX_TURNS * 2:
            conversation_history[user_id] = conversation_history[user_id][-(MAX_TURNS * 2):]

        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=conversation_history[user_id]
        )
        reply = response.content[0].text

        conversation_history[user_id].append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})

    except anthropic.RateLimitError:
        if conversation_history[user_id] and conversation_history[user_id][-1]["role"] == "user":
            conversation_history[user_id].pop()
        return jsonify({"reply": "\u0639\u0630\u0631\u0627\u064b\u060c \u0623\u0646\u0627 \u0645\u0634\u063a\u0648\u0644\u0629 \u0627\u0644\u0622\u0646. \u0623\u0631\u0633\u0644 \u0631\u0633\u0627\u0644\u062a\u0643 \u0645\u0631\u0629 \u0623\u062e\u0631\u0649 \u0628\u0639\u062f \u062b\u0627\u0646\u064a\u0629."})
    finally:
        lock.release()


@app.route('/chat', methods=['POST'])
def chat():
    return _handle_chat("claude-sonnet-4-6")


@app.route('/chat-haiku', methods=['POST'])
def chat_haiku():
    return _handle_chat("claude-haiku-4-5-20251001")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
