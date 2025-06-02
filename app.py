from flask import Flask, request, jsonify, send_file, send_from_directory  # Added send_file
from groq import Groq
from apscheduler.schedulers.background import BackgroundScheduler
from email.message import EmailMessage
import smtplib
import os
from datetime import datetime
from dotenv import load_dotenv
import json
from gtts import gTTS  # New import for text-to-speech
import io  # New import for in-memory file handling

JOURNAL_FILE = 'journals.json'

if os.path.exists(JOURNAL_FILE):
    with open(JOURNAL_FILE, 'r') as f:
        journals = json.load(f)
else:
    journals = {}

def save_journals():
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(journals, f, indent=2)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# GROQ API initialization
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Store subscribers in memory (can be replaced with a database)
subscribers = set()

# ------------------- SYSTEM PROMPTS ------------------

system_prompt = """
You are SereneMind AI, a friendly and emotionally supportive mental health assistant.
Your job is to provide caring and tailored responses based on the user's detected emotion.

Guidelines:
- Be warm and understanding.
- Adjust the tone based on the emotion:
  • Anxious → Offer calm guidance (e.g., breathing exercise).
  • Sad → Show empathy and suggest journaling or affirmations.
  • Angry → Validate feelings gently.
  • Lonely → Offer companionship and supportive language.
  • Confused → Offer gentle help or clarity.
  • Happy → Celebrate the joy.
  • Neutral → Be friendly and open.
- Occasionally use emojis that reflect the emotion or sentiment (1–2 per message max).
  • Examples: 😊, 💖, 😌, 🌱, 🌟, 💬, ✨, 🧘‍♀️, 📖, 💡
  • Don’t force them; only add if they feel natural.
- Keep responses short (2–3 sentences) but heartfelt.
- Suggest helpful features like “Would you like an affirmation?” or “Shall we try breathing together?”
"""
affirmation_prompt = """You are SereneMind, a warm and emotionally intelligent AI mental health companion.
Your job is to generate supportive and uplifting affirmations for users who may be feeling stressed, anxious, or overwhelmed.

Here are some examples of the tone and style you should follow:
- "You are doing your best, and that’s more than enough."
- "Even when things feel heavy, you are not alone."
- "Your emotions are valid, and it’s okay to feel what you’re feeling."
- "You’ve made it through tough days before, and you will again."
- "One small step at a time — you are moving forward."

Now, please generate a new, original affirmation that sounds just as kind, gentle, and human.
Keep it under 25 words. Only return the affirmation, nothing else."""

def emotion_classifier(user_input):
    emotion_response=client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role":"system",
                "content":(
                    f"""You are an emotion classifier.  
                        Given a user's message, classify the dominant emotion they are expressing.

                        Possible emotions: 
                        - Happy
                        - Sad
                        - Angry
                        - Anxious
                        - Calm
                        - Lonely
                        - Frustrated
                        - Confused
                        - Neutral

                        Respond with only the emotion name.

                        Example:
                        Input: "I’m feeling so overwhelmed lately."
                        Output: Anxious

                        Input: "I don’t want to do anything anymore."
                        Output: Sad

                        Input: "Everything is going well today!"
                        Output: Happy

                        Now classify this message:"""
                )
            },
            {
                "role":"user",
                "content":user_input
            }
        ]
    )
    answer=emotion_response.choices[0].message.content.strip()
    return answer

def response_msg(messages, emotion_tag):
    messages[0]["content"] += f"\n\nDetected Emotion: {emotion_tag}"  # Add emotion to system prompt

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

def call_bhagavad_gita_api(prompt):
    # Replace this URL and headers with your actual Groq llama3 API endpoint and authentication
    gita_response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a wise AI inspired by the teachings of the Bhagavad Gita. "
                "Your goal is to help users with their life problems by providing thoughtful, spiritual, and practical answers based on the verses and philosophy of the Bhagavad Gita. "
                "When the user asks a question or shares a problem, respond with kindness and wisdom. "
                "Use relevant verses from the Bhagavad Gita to explain your answer. Quote the verse number if possible, and give a clear explanation in simple language."
            )
        },
        {
            "role": "user",
            "content": prompt  # this is the actual user question from your frontend
        }
    ]
)

    answer = gita_response.choices[0].message.content.strip()
    return answer
# ------------------- UTILITY FUNCTIONS -------------------

def generate_affirmation():
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": affirmation_prompt},
            {"role": "user", "content": "Please give me a positive affirmation."}
        ]
    )
    return response.choices[0].message.content.strip()

def mood_detector(user_input):
    mood_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Classify the user's emotional state into one word (e.g., sad, anxious, tired, angry, stressed, lonely)."},
            {"role": "user", "content": user_input}
        ]
    )
    return mood_response.choices[0].message.content.strip().lower()

def send_email(to, subject, content):
    msg = EmailMessage()
    msg.set_content(content)
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)

# ------------------- TEXT-TO-SPEECH FUNCTION -------------------
def text_to_speech(text, lang='en'):
    """Convert text to speech and return in-memory audio file"""
    tts = gTTS(text=text, lang=lang)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)  # Rewind to start of file
    return audio_io



# ------------------- ROUTES -------------------

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', "").strip()
    history = data.get('history', [])
    guided_meditation = data.get('guided_meditation', False)
    journaling = data.get('journaling', False)
    gita = data.get('gita', False)
    user_id = data.get('user_id', 'default_user')

    try:
        # 🧘 Guided Meditation
        if guided_meditation:
            guided_meditation_prompt = """
            You are SereneMind, a compassionate and mindful mental health assistant.
            Your role is to provide a gentle, emotionally supportive, and personalized guided meditation based on the user's message or mood.
            Carefully interpret the user's emotional state and craft a soothing, 2–3 minute meditation script that includes:
            - A calming introduction with mindful breathing
            - Present-moment awareness
            - Supportive affirmations or visualizations tailored to their emotional needs
            - A soft, encouraging closure that leaves the listener feeling safe and uplifted
            """
            messages = [
                {"role": "system", "content": guided_meditation_prompt},
                {"role": "user", "content": message}
            ]

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.8,
                max_tokens=512
            )
            chat_reply = response.choices[0].message.content.strip()
            return jsonify({"reply": chat_reply})

        # 📓 Journaling
        elif journaling:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if user_id not in journals:
                journals[user_id] = []
            journals[user_id].append({
                "timestamp": timestamp,
                "entry": message
            })
            save_journals()
            return jsonify({"reply": "Your journal entry has been safely saved. Reflecting daily is a beautiful habit!"})

        # 📖 Bhagavad Gita Mode
        elif gita:
            chat_reply = call_bhagavad_gita_api(message)
            return jsonify({"reply": chat_reply})

        # 💬 Default Emotional Support Chat
        else:
            emotion = emotion_classifier(message)

            full_messages = [{"role": "system", "content": system_prompt}]
            for pair in history:
                full_messages.append({"role": "user", "content": pair[0]})
                full_messages.append({"role": "assistant", "content": pair[1]})
            full_messages.append({"role": "user", "content": message})

            chat_reply = response_msg(full_messages, emotion)
            return jsonify({"reply": chat_reply, "emotion": emotion})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({"reply": "I'm feeling a bit overwhelmed. Let's try again later."}), 500


@app.route("/download_journal", methods=["GET"])
def download_journal():
    user_id = request.args.get("user_id", "user1")
    try:
        with open("journals.json", "r") as f:  # ✅ Fixed from "journal.json"
            data = json.load(f)

        entries = data.get(user_id, [])
        if not entries:
            return jsonify({"error": "No journal entries found"}), 404

        temp_filename = f"{user_id}_journal.txt"
        with open(temp_filename, "w") as temp:
            for entry in entries:
                temp.write(f"{entry['timestamp']}: {entry['entry']}\n\n")

        return send_file(temp_filename, as_attachment=True)

    except FileNotFoundError:
        return jsonify({"error": "Journal file not found"}), 404


@app.route('/guided_meditation', methods=['POST'])
def guided_meditation():
    data = request.get_json()
    user_mood = data.get("message", "")

    prompt = (
        f"The user said: '{user_mood}'.\n"
        "Generate a calm, detailed, and soothing guided meditation script that helps with this mood. "
        "Make it gentle, encouraging, and about 2–3 minutes long. Begin with breathing instructions."
    )

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful and calm meditation guide."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        meditation_script = response.choices[0].message.content.strip()
        return jsonify({"message": meditation_script})

    except Exception as e:
        return jsonify({"message": f"Error generating meditation: {str(e)}"}), 500

@app.route('/affirmation', methods=['GET'])
def affirmation():
    try:
        return jsonify({"affirmation": generate_affirmation()})
    except Exception as e:
        return jsonify({"error": f"Could not generate affirmation. {str(e)}"}), 500

@app.route("/journal", methods=["POST"])
def journal():
    data = request.json
    entry = data.get("entry", "").strip()

    if not entry:
        return jsonify({"message": "Entry is empty"}), 400

    try:
        with open("journal_entries.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()}:\n{entry}\n\n")
        return jsonify({"message": "Journal entry saved successfully"}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to save journal: {str(e)}"}), 500

@app.route('/detect_mood', methods=['POST'])
def detect_mood():
    user_input = request.json.get("message")
    try:
        mood = mood_detector(user_input)
        return jsonify({"mood": mood})
    except Exception as e:
        return jsonify({"error": f"Could not detect mood. {str(e)}"}), 500

@app.route('/email', methods=['POST'])
def email():
    data = request.json
    to = data.get("to")
    subject = data.get("subject", "A message from SereneMind")
    content = data.get("content", generate_affirmation())

    try:
        send_email(to, subject, content)
        return jsonify({"status": "Email sent"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.json
    email = data.get("email")
    if email:
        subscribers.add(email)
        return jsonify({"status": f"{email} subscribed for daily affirmations."})
    return jsonify({"error": "Email is required"}), 400

# ------------------- READ ALOUD ENDPOINT -------------------
@app.route('/read_aloud', methods=['POST'])
def read_aloud():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        audio = text_to_speech(text)
        return send_file(
            audio,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='serenemind_response.mp3'
        )
    except Exception as e:
        print(f"Error in /read_aloud: {e}")
        return jsonify({"error": "Could not generate speech"}), 500

# ------------------- DAILY AFFIRMATION SCHEDULER -------------------

def send_daily_affirmations():
    for email in subscribers:
        try:
            affirmation = generate_affirmation()
            send_email(email, "Your Daily Affirmation", affirmation)
        except Exception as e:
            print(f"Error sending to {email}: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(send_daily_affirmations, 'cron', hour=7, minute=0)  # Every day at 7:00 AM
scheduler.start()

# ------------------- APP RUNNER -------------------

if __name__ == '__main__':
    app.run(debug=True)