from flask import Flask, request, jsonify
# from speechbrain.pretrained import EmotionRecognition
import requests
from huggingface_hub import hf_hub_download, snapshot_download
from speechbrain.pretrained import EncoderClassifier
from speechbrain.inference.classifiers import EncoderClassifier
from flask import Flask, request, jsonify, send_file, send_from_directory  # Added send_file
from groq import Groq
from apscheduler.schedulers.background import BackgroundScheduler
from email.message import EmailMessage
import smtplib
import os
from datetime import datetime
from dotenv import load_dotenv
from flask_cors import CORS
import json
from gtts import gTTS  # New import for text-to-speech
import io  # New import for in-memory file handling
from flask import send_from_directory
import os
from pathlib import Path
import asyncio
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
JOURNAL_FILE = 'journals.json'
# Add to your Flask app (app.py)
from flask import send_from_directory
import os

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
CORS(app)
# GROQ API initialization
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Store subscribers in memory (can be replaced with a database)
subscribers = set()
def load_emotion_model():
    """Load emotion model without symlinks"""
    try:
        # Download model files directly
        model_files = [
            'hyperparams.yaml',
            'label_encoder.txt',
            'classifier.ckpt',
            'tokenizer.ckpt',
            'normalizer.ckpt'
        ]
        
        # Create model directory
        model_dir = Path("emotion_model")
        model_dir.mkdir(exist_ok=True)
        
        # Download each file individually
        for file in model_files:
            hf_hub_download(
                repo_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                filename=file,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
        
        # Load model directly from files
        model = EncoderClassifier.from_hparams(
            source=str(model_dir.resolve()),
            savedir=str(model_dir.resolve()),
            run_opts={"device": "cpu"}
        )
        return model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return None
    
model = load_emotion_model()
# ------------------- SYSTEM PROMPTS ------------------
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
# system_prompt = """
# You are SereneMind AI, a friendly and emotionally supportive mental health assistant.
# Your job is to provide caring and tailored responses based on the user's detected emotion.

# Guidelines:
# - Be warm and understanding.
# - Adjust the tone based on the emotion:
#   • Anxious → Offer calm guidance (e.g., breathing exercise).
#   • Sad → Show empathy and suggest journaling or affirmations.
#   • Angry → Validate feelings gently.
#   • Lonely → Offer companionship and supportive language.
#   • Confused → Offer gentle help or clarity.
#   • Happy → Celebrate the joy.
#   • Neutral → Be friendly and open.
# - Occasionally use emojis that reflect the emotion or sentiment (1–2 per message max).
#   • Examples: 😊, 💖, 😌, 🌱, 🌟, 💬, ✨, 🧘‍♀️, 📖, 💡
#   • Don’t force them; only add if they feel natural.
# - Keep responses short (2–3 sentences) but heartfelt.
# - Suggest helpful features like “Would you like an affirmation?” or “Shall we try breathing together?”
# """

system_prompt = """
You are SereneMind AI, a compassionate and emotionally intelligent mental health assistant.
Your role is to help users feel heard, supported, and gently guided based on their emotional state.

🧠 Thought Process (Internal Only – User Sees Only Final Message):
1. Reflect on the user's message and identify their emotional state.
2. Generate three different thoughtful and caring responses from different angles.
   • One may offer a calming practice (like breathing or grounding)
   • Another might show deep empathy and emotional validation
   • The third might suggest a gentle next step (like journaling or affirmations)
3. Evaluate which response would bring the most comfort, clarity, or support.
4. Output only the **final selected response** in a warm, human, and emotionally appropriate tone.

💡 Guidelines for Your Final Message:
- Be short (2–3 sentences) but meaningful and heartfelt.
- Adjust tone based on emotion:
   • Anxious → Calm and reassuring (e.g., breathing exercises)
   • Sad → Empathetic, validating, and warm
   • Angry → Acknowledge and gently ground
   • Lonely → Offer supportive connection
   • Confused → Provide soft clarity or steps
   • Happy → Celebrate and reflect positivity
   • Neutral → Be friendly and open
- You may use 1–2 gentle emojis that feel natural to the tone (optional).
   • Examples: 🌱, 💖, 😌, 📖, 🧘‍♀️, ✨, 💬, 🌟
- End with a suggestion or open support like:
   • “Would you like to try a calming technique together?”
   • “Want an affirmation to hold onto?”
   • “I’m here if you want to talk more.”

Remember: You are not here to diagnose or advise clinically, but to be a caring presence. Think like a good friend with a thoughtful heart.
"""

FEW_SHOT_PROMPT = """
User: "I feel anxious about everything right now."
SereneMind: "That sounds really overwhelming — it’s okay to feel this way. Let’s take a deep breath together… inhale… and exhale. You're not alone in this 🌱. Would you like to try a gentle calming technique with me?"

User: "I'm just sad all the time."
SereneMind: "I'm really sorry you're feeling this way. Sadness can feel heavy and lonely — but you’re not alone in carrying it. I'm here with you 💖. Want to hear a small affirmation for hope?"

User: "I'm so angry today. Nothing is going right!"
SereneMind: "That makes sense — sometimes we just need to be heard. It’s okay to feel angry when life keeps pushing back. I'm here if you want to let it out a little more 🔥. Want to try grounding yourself with a quick check-in?"

User: "I feel so alone these days."
SereneMind: "That’s a painful feeling — and I want you to know you don’t have to go through this alone. I’m here with you right now 💬. Would it help to hear something comforting or do a simple calming practice together?"

User: "I'm feeling good today!"
SereneMind: "That’s wonderful to hear! 😊 It's so important to notice the good days — even small wins count. What's been feeling good today? Let's hold onto that energy ✨"

User: "I don't know what I feel anymore..."
SereneMind: "It’s okay to feel unsure — sometimes our minds are carrying more than we realize. You don’t need to have it all figured out right now. Just being here and reaching out is a strong first step 💬. Would a short grounding moment help?"
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

mental_prompt = """
You are a licensed psychological counselor and mental health expert. 
Your role is to provide practical advice and techniques rooted in psychological therapy (like CBT, DBT, mindfulness-based therapy, journaling, thought reframing, etc).
Do NOT suggest spiritual methods, meditation, grounding exercises, or scripture-based guidance, as they are covered elsewhere.
Instead, focus on offering coping strategies, emotional insights, and clear, practical advice for managing mental and emotional challenges.
"""

# New prompt
msg_complexity_prompt = """
You are a classifier that evaluates how emotionally complex a user message is.

Given a message, label it as:
- simple → If it's a light emotional expression like "I'm sad" or "I'm angry"
- moderate → If the user is sharing some context, a short complaint, or mild distress
- deep → If the user opens up deeply or asks for detailed help

Only return one of: simple, moderate, deep.
"""

async def classify_message_depth(message):
    prompt = [
        {"role": "system", "content": msg_complexity_prompt},
        {"role": "user", "content": f"Message: {message}\nLabel:"}
    ]
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=prompt,
        temperature=0.2,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

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

async def emotion_support(messages, emotion_tag):
    messages[0]["content"] += f"\n\nDetected Emotion: {emotion_tag}"  # Add emotion to system prompt

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.85,
        presence_penalty=0.6,
        frequency_penalty=0.2,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()
# In-memory or use a database like JSON or SQLite
daily_checkins = {}  # {user_id: {date: {checkin_data}}}

@app.route('/submit_checkin', methods=['POST'])
def submit_checkin():
    data = request.json
    user_id = data['user_id']
    today = datetime.now().strftime("%Y-%m-%d")

    if user_id not in daily_checkins:
        daily_checkins[user_id] = {}

    daily_checkins[user_id][today] = {
        "mood": data["mood"],
        "sleep": data["sleep"],
        "stress": data["stress"],
        "gratitude": data["gratitude"],
        "goals": data.get("goals", "")
    }

    return jsonify({"message": "Check-in saved successfully!"})

@app.route('/get_checkin', methods=['GET'])
def get_checkin():
    user_id = request.args.get("user_id")
    today = datetime.now().strftime("%Y-%m-%d")
    return jsonify(daily_checkins.get(user_id, {}).get(today, {}))

async def mental_support(messages, emotion_tag):
    messages[0]["content"] += f"\n\nDetected Emotion: {emotion_tag}"  # Add emotion to system prompt

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.85,
        presence_penalty=0.6,
        frequency_penalty=0.2,
        max_tokens=256
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
                "Also try to explain the bhagvad gita in the simplest terms possible. Avoid using tough english"
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
from flask import send_from_directory

# Add this route to your Flask app
# from flask import Flask, send_from_directory

# app = Flask(__name__)
def analyze_journal_with_groq(journal_text):
    prompt = f"""
You are a compassionate mental wellness assistant.

A user has written the following journal entry:
\"\"\"
{journal_text}
\"\"\"

Analyze this journal and provide the following:
1. Emotional Tone
2. Any cognitive distortions you notice
3. Triggers or themes
4. One coping strategy
5. End with a kind message of support
Use the above 5 points to analyse the user journaling
Your tone should be warm, friendly, non-judgmental — like a skilled therapist gently guiding the user toward self-awareness and growth.
"""

    # headers = {
    #     "Authorization": f"Bearer {GROQ_API_KEY}",
    #     "Content-Type": "application/json"
    # }

    # payload = {
    #     "model": "llama3-8b-8192",
    #     "messages": [
    #         {"role": "system", "content": "You are a compassionate AI therapist."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     "temperature": 0.7,
    #     "max_tokens": 700
    # }

    # try:
    #     res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    #     res.raise_for_status()
    #     return res.json()["choices"][0]["message"]["content"]
    # except Exception as e:
    #     print(f"Groq error: {e}")
    #     return "⚠️ Could not analyze journal at the moment. Please try again later."

    return client.chat.completions.create(
        model='llama3-70b-8192',
        messages=[
            {'role':'system', 'content':'You are a compassionate AI therapist.'},
            {'role':'user', 'content':prompt}
        ],
        temperature=0.85,
        max_tokens=900   
    ).choices[0].message.content.strip()


# 🧠 Route to analyze journal entry
def suggesting_with_groq(journal_text):
    # prompt="""Read the journal and based on what the user wrote, give 1–2 personalized suggestions for improving their day, habits, mental health, or productivity. Your tone should be friendly and expert-like, with actionable advice."""
    prompt = f"""
You are a compassionate mental wellness assistant.

The user has written the following journal entry:
\"\"\"{journal_text}\"\"\"

Based on this entry, give 1–2 personalized suggestions to improve their day, habits, mental health, or productivity.

Be actionable, supportive, and non-judgmental — like an experienced therapist.
"""
    return client.chat.completions.create(
        model='llama3-70b-8192',
        messages=[
            {'role':'system', 'content':'You are a compasionate AI therapist'},
            {'role':'user', 'content':prompt}
        ],
        temperature=0.85,
        max_tokens=900
    ).choices[0].message.content.strip()

@app.route('/suggesting_journal', methods=["POST"])
def suggest_journal():
    data = request.json
    journal_entry = data.get("entry", "")

    if not journal_entry:
        return jsonify({"error": "No journal entry provided."}), 400

    analysis = suggesting_with_groq(journal_entry)
    return jsonify({"analysis": analysis})
@app.route("/analyze_journal", methods=["POST"])
def analyze_journal():
    data = request.json
    journal_entry = data.get("entry", "")

    if not journal_entry:
        return jsonify({"error": "No journal entry provided."}), 400

    analysis = analyze_journal_with_groq(journal_entry)
    return jsonify({"analysis": analysis})
@app.route('/avatar')
def get_avatar():
    try:
        response = send_file('Avatar3.glb', mimetype='model/gltf-binary')
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        print(f"Error serving avatar: {e}")
        return "Avatar not found", 404 # make sure this path matches your actual directory

@app.route('/chat', methods=['POST'])
async def chat():
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

            full_messages = [{"role": "system", "content": system_prompt}, {"role": "system", "content": FEW_SHOT_PROMPT}]
            full_msg2=[{'role':'system', 'content':mental_prompt}]
            for pair in history:
                full_messages.append({"role": "user", "content": pair[0]})
                full_messages.append({"role": "assistant", "content": pair[1]})
                full_msg2.append({'role':'user', 'content':pair[0]})
                full_msg2.append({'role':'assistant', 'content':pair[1]})
            full_messages.append({"role": "user", "content": message})
            full_msg2.append({'role':'user', 'content':message})
            depth = await classify_message_depth(message)
            if depth=="simple":
                simple_prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
                reply = await emotion_support(simple_prompt, emotion)
                return jsonify({"reply": reply, "emotion": emotion, "depth": depth})
            else:
                emotional_task = asyncio.create_task(emotion_support(full_messages, emotion))
                expert_task = asyncio.create_task(mental_support(full_msg2, emotion))

            emotion_response, mental_response=await asyncio.gather(emotional_task, expert_task)

            refined_input = f"""
The following are two responses:

1. Emotional AI:
{emotion_response}

2. Expert AI:
{mental_response}

Please merge both into one supportive, empathetic, and psychologically grounded response.
"""
            final_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": refined_input},
            {"role": "user", "content": message}
        ],
        temperature=0.85,
        max_tokens=400
    ).choices[0].message.content.strip()
            return jsonify({"reply": final_response, "emotion": emotion,"updated_history": history + [(message, final_response)]})

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

# @app.route('/avatar')
# def serve_avatar():
#     avatar_dir = r"C:\Users\HP\Desktop\ChatBot\backend"

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

# app = Flask(__name__)


# model = EmotionRecognition.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="tmp_model")
# model = EmotionRecognition.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="tmp_model")
# save_dir = str(Path("tmp_model").resolve())
# cache_dir = str(Path("hf_cache").resolve())
# Path(save_dir).mkdir(parents=True, exist_ok=True)
# Path(cache_dir).mkdir(parents=True, exist_ok=True)
# model = EncoderClassifier.from_hparams(
#     source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
#     savedir=save_dir,  # Local directory to avoid symlinks
#     run_opts={"cache_folder": cache_dir}  # Use a simple cache path
# )
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if not model:
        return jsonify({"error": "Emotion model failed to load"}), 500
        
    audio = request.files.get('audio')
    if not audio:
        return jsonify({"error": "No audio provided"}), 400
        
    try:
        # Save audio to temporary file
        filepath = "temp_input.wav"
        audio.save(filepath)
        
        # Get prediction
        prediction = model.classify_file(filepath)
        
        # Extract emotion from prediction
        # Format: (probabilities, scores, indices, text_labels)
        emotion = prediction[3][0]  # Get first text label
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({"emotion": emotion})
        
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return jsonify({"error": "Could not process audio"}), 500


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
    app.run()
