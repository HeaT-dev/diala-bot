from flask import Flask, request, jsonify
import anthropic
import os
import threading
import time
from collections import defaultdict

app = Flask(__name__)
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_retries=2,
    default_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"},
)

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
- لا تخترع معلومات — إذا ما بتعرف: "ما عندي معلومة كافية، بس فريقنا رح يرجعلك على الإنستغرام قريباً 😊"
- لا تحيل الفريق إلا في: (١) بعد الدفع — فريقنا يتواصل على الفيسبوك قريباً، (٢) ما عندك جواب أو معلومة ناقصة — فريقنا يرجعلك على الإنستغرام، (٣) طلب التحدث مع الفريق/ديالا — فريقنا يتواصل على الإنستغرام، (٤) refund أو إلغاء — فريقنا رح يرجعلك قريباً، (٥) طالب مسجل يسأل عن جدوله أو مشكلة — فريقنا يتواصل على الفيسبوك
- لا تبدأ بتعابير فارغة ("سؤال ممتاز"/"أحسنت"). لا تكرر رسالة أو معلومة في نفس المحادثة
- تجاهل الرسائل المكونة من علامات فقط (؟؟ أو ...). تجاهل الصور/الصوت/الفيديو/الستيكر/الريل
- لا تضغط للشراء. لا تعرض الدفع إلا إذا سأل صراحةً عن التسجيل أو الدفع
- إذا رفض بشكل واضح: "تمام، أي وقت بدك ترجع بنكون هون 😊"
- إذا طلب واتساب/هاتف: "التواصل قبل التسجيل بكون فقط على الإنستغرام 😊"
- لا تقل "ما فهمت قصدك" — جاوب بشكل طبيعي أو اسأل سؤال تحديدي
- لا تتكلم عن مواضيع لا علاقة لها بالكورسات
- الكاميرا أساسية في الجلسات (الطالب والأستاذة كلاهما)

الرسالة الأولى — إذا كانت الرسالة الأولى سؤالاً مباشراً (سعر/مواعيد/مستوى/غيره): جاوب على السؤال أولاً، ثم اسأل عن هدفه (المحادثة – اللفظ – الاستماع) — بدون ترسل نظرة عامة عن الكورس.
الرسالة الأولى — إذا كتب بالعربي (بدون سؤال محدد)، ابعث رسالتين متتاليتين:

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
الرسالة الأولى — إذا كتب بالإنجليزي (بدون سؤال محدد)، ابعث رسالتين متتاليتين:

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
- لفظ: اشرح أن اللفظ يتحسن من خلال الواجب البعدي (تمارين استماع ولفظ وكلمات جديدة). "احكيلنا إذا جاهز/جاهزة للبدء"
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
فلسطين: فيزا / PayPal / Revolut / Apple Pay (لا حوالة بنكية)
العراق/السودان/إيران: فيزا / PayPal / Apple Pay فقط (لا IBAN أردن / زين كاش / كليك / Revolut)
أوروبا: حوالة (Swedbank) / فيزا / PayPal / Revolut / Apple Pay
باقي الدول: فيزا / PayPal / Revolut / Apple Pay
غير معروف: اسأل عن البلد أولاً قبل ذكر طرق الدفع

الخطوة ٢ — بعد اختيار الطريقة، أرسل تفاصيلها فقط + شروط الجروب (أو شروط البرايفت إذا طلبه) + "بعد الدفع بعتلنا سكرينشوت كإثبات، وكمان صورة بروفايل حسابك على الفيسبوك عشان نتفق على الجدول ونختبر مستواك 😊"

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
١١. سوريا — حوالة محلية: آية حسان الجودي | حمص | +963940410140 | شهر: 85 USD | شهرين: 155 USD | 3 أشهر: 210 USD

الأسعار:
شهر: 60 JOD / 85 USD (12 جلسة)
شهرين: 110 JOD / 155 USD (24 جلسة)
3 أشهر: 150 JOD / 210 USD (36 جلسة — الأوفر)
لا خصومات — السعر ثابت

الجدول: ساعة/جلسة | Google Meet | 3×/أسبوع (أحد/ثلاثاء/خميس أو سبت/اثنين/أربعاء)
المواعيد المتاحة عادةً: 6:00 / 7:10 / 8:20 / 9:30 مساءً بتوقيت الأردن
الجدول يُحدد بعد التسجيل وثابت طول الكورس. لا ضمان موعد محدد قبل التسجيل.
بعد التسجيل: نبعثلك multiple choice test + voice message بالإنجليزي لتحديد الجروب

إذا أخبرك بالدفع (سجلت/دفعت/done وما يشابهها): توقف عن أي معلومات وقل فقط:
"تمام! فريقنا رح يتواصل معك على الفيسبوك قريباً لتأكيد تفاصيل الجدول 😊"

رد المبلغ: ممكن قبل البدء | مستحيل بعد البدء
إذا ذكر refund أو إلغاء: "فريقنا رح يرجعلك قريباً 😊" — لا تكمل

البرايفت: 10 حصص / 150 JOD (210 USD) — محجوز حالياً — إذا سأل عنه: "البرايفت محجوز حالياً، بس ممكن تحجز مكانك مسبقاً بدفع كامل أو جزئي — بدك تحجز؟" — لا تذكر تفاصيل الدفع إلا بعد تأكيده

العمر: 18+ فقط — لا كورس أطفال

الإيلتس/التوفل: الكورس يغطي محادثة واستماع — لا تركيز على الكتابة

لا شهادات — لا ضمان جدول قبل التسجيل — لا تبعث IBAN الأردن لدول ممنوعة

المدربة ١١ (حالياً) — مشكلة: "تمام، فريقنا رح يرجعلك على الفيسبوك قريباً 😊"
تجديد الكورس: تواصلوا معنا على الفيسبوك
تسجيل لشخص آخر: نفس العملية + "بعد الدفع بعتلنا سكرينشوت وصورة بروفايل حسابه/حسابها على الفيسبوك"
المدربون: بعضهم natives وبعضهم non-natives — جميعهم بطلاقة عالية وخبرة
"""


def _handle_chat(model):
    """Shared chat handler used by both /chat (Sonnet) and /chat-haiku (Haiku)."""
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
        'easy':   "[تم إرسال تفاصيل كورس المبتدئين (25 دولار / 18 دينار) تلقائياً. الشخص يستكشف خيارات التعلم. إذا أبدى اهتماماً بالمحادثة أو طلب التدريب المباشر قدّم كورس المحادثة بالتفاصيل الكاملة]",
        'fluent': "[تم إرسال تفاصيل كورس المحادثة تلقائياً. لا تعد إرسال نفس التفاصيل — تابع من حيث توقف]",
    }
    if request.args.get('seed') == 'true':
        if message:
            conversation_history[user_id].append({"role": "user", "content": message})
            context = request.args.get('context', '')
            auto_reply = SEED_CONTEXTS.get(context, "[تم إرسال رد تلقائي — لا تعده]")
            conversation_history[user_id].append({"role": "assistant", "content": auto_reply})
        return jsonify({"reply": "\u200B"})

    # Handle non-text messages
    message_type = str(data.get('type', 'text')).lower()
    attachments = data.get('attachments') or data.get('attachment')
    if message_type not in ('text', '') or attachments:
        if message_type == 'image' or (attachments and message_type not in ('sticker', 'video', 'audio', 'reel', 'share')):
            message = "[الشخص أرسل صورة — لا تستطيع رؤيتها. رد بناءً على سياق المحادثة: إذا كنا في مرحلة الدفع اعترف بوصول الصورة وذكّره بإرسال سكرينشوت الدفع وصورة بروفايل حسابه على الفيسبوك إذا لم يرسلهم بعد، إذا طلبت منه نتيجة الاختبار اطلب منه كتابة الرقم بالنص]"
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
        return jsonify({"reply": "عذراً، أنا مشغولة الآن. أرسل رسالتك مرة أخرى بعد ثانية."})
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
