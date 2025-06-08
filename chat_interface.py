import gradio as gr
import requests
import random
API_BASE = "http://127.0.0.1:5000"  # Adjust if running elsewhere


emojis = ["😀", "😂", "😢", "😍", "😡", "😎", "👍", "🙏", "💯", "💖", "🤯", "🎉", "😴", "👀"]
def handle_chat(message, history, guided=False, journaling=False, user_id="user1", gita=False):
    try:
        payload = {
            "message": message,
            "history": history,
            "guided_meditation": guided,
            "journaling": journaling,
            "user_id": user_id,
            "gita":gita
        }
        res = requests.post(f"{API_BASE}/chat", json=payload)
        data = res.json()
        reply = data.get("reply", "Something went wrong.")
        updated_history=data.get("updated_history", history)
        expression = "neutral"
        positive_triggers = ["good", "great", "happy", "joy", "love", "thanks"]
        negative_triggers = ["sad", "bad", "angry", "stress", "anxious", "depressed"]
        
        if any(trigger in reply.lower() for trigger in positive_triggers):
            expression = "happy"
        elif any(trigger in reply.lower() for trigger in negative_triggers):
            expression = "concerned"
        history.append((message, reply))
        return updated_history,updated_history, expression
    except Exception as e:
        return history, history, "confused"
    
import os


def save_user_voice(audio_file):
    if audio_file is None:
        return "No audio received.", "😐"

    try:
        # Save the file locally if needed
        saved_path = "user_voice.wav"
        os.rename(audio_file, saved_path)

        # Send to backend for emotion detection
        with open(saved_path, "rb") as f:
            files = {"audio": f}
            res = requests.post(f"{API_BASE}/detect_emotion", files=files)
        
        if res.status_code == 200:
            emotion = res.json().get("emotion", "Unknown")
            return f"🎧 Emotion detected: {emotion}", emotion
        else:
            return "❌ Could not detect emotion", "😐"
    except Exception as e:
        print("Error in save_user_voice:", e)
        return "❌ Error occurred during voice processing", "😐"

def build_prompt(history, journal):
    full_history = "\n".join(history)
    full_journal = "\n".join(journal)

    return f"""
You are SereneMind, a mental health assistant.

Analyze the user's emotional state from their past conversations and journal entries. Then give personalized mental wellness suggestions including:
1. A grounding technique
2. A type of meditation
3. A Bhagavad Gita verse (brief)
4. A short affirmation

Please be concise but complete. End your response with an encouraging closing sentence.

### User Chat History:
{full_history}

### Journal Entries:
{full_journal}

Now give your suggestions in a clean format.
"""

def react(emoji, history):
    if not history:
        history = []
    history.append(("User reacted with:", emoji))
    return history
grounding_techniques = [
    "5-4-3-2-1 Technique: Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
    "Take 10 deep breaths: Inhale slowly for 4 seconds, hold for 4 seconds, and exhale for 4 seconds.",
    "Clench your fists tightly for 5 seconds, then release and feel the tension melt away.",
    "Focus on your feet on the ground and notice the sensation of contact with the floor.",
    "Describe your surroundings in detail to yourself — colors, shapes, textures."
]
import requests

def get_suggestions_groq(prompt):
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",  # or "mixtral-8x7b-32768"
        "messages": [
            {"role": "system", "content": "You are a compassionate AI therapist."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }

    response = requests.post(f"https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content']

def get_grounding_technique():
    return random.choice(grounding_techniques)
def get_affirmation():
    res = requests.get(f"{API_BASE}/affirmation")
    return res.json().get("affirmation", "Affirmation could not be fetched.")

def read_aloud(text):
    try:
        response = requests.post(f"{API_BASE}/read_aloud", json={"text": text})
        with open("temp_audio.mp3", "wb") as f:
            f.write(response.content)
        return "temp_audio.mp3"
    except Exception:
        return None

def fetch_journal_file(user_id="user1"):
    try:
        res = requests.get(f"{API_BASE}/download_journal", params={"user_id": user_id})
        if res.status_code == 200:
            with open("downloaded_journal.txt", "wb") as f:
                f.write(res.content)
            return "downloaded_journal.txt"
        else:
            print("Error from server:", res.text)
            return None
    except Exception as e:
        print("Download failed:", e)
        return None

with gr.Blocks(title="SereneMind AI") as demo:
    gr.Markdown("## 🌿 SereneMind: Your Mental Health Companion")

    with gr.Row():
        chat_box = gr.Chatbot(label="SereneMind Chat")
        with gr.Column():
            user_input = gr.Textbox(
    placeholder="How are you feeling today?",
    label="Your message",
    lines=3,
    max_lines=3,
    # scroll=True
)
            with gr.Row():
                emoji_dropdown = gr.Dropdown(
                    choices=emojis,
                    label="Add Emoji",
                    value=None,
                    interactive=True,
                    scale=3
                )
                add_emoji_btn = gr.Button("Add", scale=1)
                
                # Function to add selected emoji
                def add_emoji(emoji, current_text):
                    if emoji:
                        return current_text + emoji
                    return current_text
                
                add_emoji_btn.click(
                    add_emoji,
                    inputs=[emoji_dropdown, user_input],
                    outputs=user_input
                )
            guided_toggle = gr.Checkbox(label="Guided Meditation")
            journal_toggle = gr.Checkbox(label="Journal Entry")
            gita_toggle = gr.Checkbox(label="Ask Bhagavad Gita")
            send_button = gr.Button("Send")
            clear_button = gr.Button("Clear Conversation")
            download_btn = gr.Button("Download My Journal")
            download_link = gr.File(label="Your Journal File")
    user_id_input = gr.Textbox(label="User ID", value="user1") 
    download_btn.click(fetch_journal_file, inputs=[user_id_input], outputs=download_link)
    # send_button.click(
    #     handle_chat,
    #     inputs=[user_input, chat_box, guided_toggle, journal_toggle,gr.State(value="user1"), gita_toggle],
    #     outputs=[chat_box, chat_box]
    # )

    expression_output = gr.Textbox(label="Avatar Expression", interactive=False)

    send_button.click(
        handle_chat,
        inputs=[user_input, chat_box, guided_toggle, journal_toggle, user_id_input, gita_toggle],
        outputs=[chat_box, chat_box, expression_output]
)

    clear_button.click(lambda: [], None, chat_box)

    with gr.Accordion("🧘 Affirmation & Read-Aloud", open=False):
        affirmation_btn = gr.Button("Give me an affirmation")
        affirmation_output = gr.Textbox(label="Affirmation")
        tts_audio = gr.Audio(label="Listen to it")

        affirmation_btn.click(get_affirmation, outputs=affirmation_output)
        affirmation_output.change(read_aloud, inputs=affirmation_output, outputs=tts_audio)

        with gr.Accordion("🎤 Speak to Detect Emotion", open=False):
            # voice_input = gr.Audio(source="microphone", type="filepath", label="🎙️ Record Your Voice")
            voice_input = gr.Audio(type="filepath", label="🎙️ Record Your Voice")
            detect_voice_btn = gr.Button("🔍 Detect Emotion from Voice")
            voice_emotion_output = gr.Textbox(label="🎭 Detected Emotion")

            detect_voice_btn.click(
                fn=save_user_voice,
                inputs=[voice_input],
                outputs=[voice_emotion_output, expression_output]  # expression_output can trigger avatar reaction
        )


    with gr.Accordion("🧠 Grounding Techniques", open=False):
        grounding_btn = gr.Button("Give me a grounding technique")
        grounding_output = gr.Textbox(label="Grounding Technique", interactive=False)
        grounding_btn.click(get_grounding_technique, outputs=grounding_output)

    with gr.Accordion("🧠 Personalized AI Suggestions", open=False):
        suggest_btn = gr.Button("🧠 Get AI Suggestions")
        suggest_output = gr.Textbox(label="AI Suggestion", lines=6)

        suggest_btn.click(
            fn=lambda history, journal_flag: get_suggestions_groq(
                build_prompt([f"{h[0]} → {h[1]}" for h in history], ["User enabled journaling." if journal_flag else ""])
        ),
            inputs=[chat_box, journal_toggle],
            outputs=suggest_output
        )
    
    with gr.Accordion("🗓️ Daily Wellness Check-In", open=False):
        mood_slider = gr.Slider(1, 10, label="Mood (1-10)")
        sleep_slider = gr.Slider(1, 10, label="Sleep Quality (1-10)")
        stress_slider = gr.Slider(1, 10, label="Stress Level (1-10)")
        gratitude_box = gr.Textbox(label="🙏 Gratitude for Today", placeholder="I'm grateful for...")
        goals_box = gr.Textbox(label="🎯 Any goals?", placeholder="e.g., Take a walk, read a book...")

        submit_checkin_btn = gr.Button("📥 Submit Check-In")
        checkin_status = gr.Textbox(label="Status")

        def submit_checkin(mood, sleep, stress, gratitude, goals, user_id):
            payload = {
                "mood": mood,
             "sleep": sleep,
            "stress": stress,
            "gratitude": gratitude,
            "goals": goals,
            "user_id": user_id
        }
            res = requests.post(f"{API_BASE}/submit_checkin", json=payload)
            return res.json().get("message", "Something went wrong.")

        submit_checkin_btn.click(
            submit_checkin,
            inputs=[mood_slider, sleep_slider, stress_slider, gratitude_box, goals_box, user_id_input],
            outputs=checkin_status
        )

    with gr.Accordion("📓 Smart Journaling", open=False):
        journal_input = gr.Textbox(
        label="📝 Write your journal entry",
        placeholder="Reflect on your thoughts, feelings, or anything you want to express...",
        lines=6
    )
        analyze_journal_btn = gr.Button("✨ Analyze My Journal")
        journal_analysis_output = gr.Textbox(label="🧠 AI Reflection", lines=6)
        suggestion_journal_btn = gr.Button("✨ Give me a Suggestion or improvement")
        suggestion_analysis_output = gr.Textbox(label="🧠 AI Reflection", lines=6)
        download_journal_btn = gr.Button("📥 Download Entry")
        download_journal_file = gr.File(label="📄 Saved Journal")

        def analyze_journal(entry, user_id):
            if not entry.strip():
                return "Please write something in your journal."

        # You can use your Groq API or backend here
            try:
                payload = {
                    "entry": entry,
                    "user_id": user_id
                }
                res = requests.post(f"{API_BASE}/analyze_journal", json=payload)
                if res.status_code == 200:
                    return res.json().get("analysis", "No analysis returned.")
                else:
                    return "❌ Could not analyze journal. Server error."
            except Exception as e:
                print("Error in analyze_journal:", e)
                return "⚠️ An error occurred while analyzing."
            
        def suggesting(entry, user_id):
            if not entry.strip():
                return "Please write something in your journal."

        # You can use your Groq API or backend here
            try:
                payload = {
                    "entry": entry,
                    "user_id": user_id
                }
                res = requests.post(f"{API_BASE}/suggesting_journal", json=payload)
                if res.status_code == 200:
                    return res.json().get("analysis", "No analysis returned.")
                else:
                    return "❌ Could not suggest anything from the journal. Server error."
            except Exception as e:
                print("Error in analyze_journal:", e)
                return "⚠️ An error occurred while analyzing."
            
        def save_journal_to_file(entry, user_id):
            filename = f"{user_id}_journal_entry.txt"
            with open(filename, "w") as f:
                f.write(entry)
            return filename

        analyze_journal_btn.click(analyze_journal, inputs=[journal_input, user_id_input], outputs=journal_analysis_output)
        # suggest_btn.click(suggesting_journal, inputs=)
        suggestion_journal_btn.click(suggesting, inputs=[journal_input, user_id_input], outputs=suggestion_analysis_output)
        download_journal_btn.click(save_journal_to_file, inputs=[journal_input, user_id_input], outputs=download_journal_file)



if __name__ == "__main__":
    demo.launch()
