import gradio as gr
import requests

API_BASE = "http://127.0.0.1:5000"  # Adjust if running elsewhere

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
        history.append((message, reply))
        return history, history
    except Exception as e:
        return history, history + [("Error", str(e))]
    
import random
emojis = ["😀", "😂", "😢", "😍", "😡", "😎", "👍", "🙏", "💯", "💖", "🤯", "🎉", "😴", "👀"]

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
    #         emoji_menu = gr.Radio(
    #     choices=emojis,
    #     value="😀",
    #     visible=False,
    #     label="Select an emoji",
    #     elem_id="emoji_menu"
    # )

    # # Toggle emoji menu visibility
    # def toggle_emoji_menu():
    #     return gr.update(visible=True)
    
    # emoji_btn.click(
    #     fn=toggle_emoji_menu,
    #     outputs=emoji_menu
    # )

    # Insert selected emoji into textbox
    # def insert_emoji(emoji, current_text):
    #     return current_text + emoji
    
    # emoji_menu.change(
    #     fn=insert_emoji,
    #     inputs=[emoji_menu, user_input],
    #     outputs=user_input
    # )




    # state = gr.State([])
    user_id_input = gr.Textbox(label="User ID", value="user1") 
    download_btn.click(fetch_journal_file, inputs=[user_id_input], outputs=download_link)
    send_button.click(
        handle_chat,
        inputs=[user_input, chat_box, guided_toggle, journal_toggle,gr.State(value="user1"), gita_toggle],
        outputs=[chat_box, chat_box]
    )

    clear_button.click(lambda: [], None, chat_box)

    with gr.Accordion("🧘 Affirmation & Read-Aloud", open=False):
        affirmation_btn = gr.Button("Give me an affirmation")
        affirmation_output = gr.Textbox(label="Affirmation")
        tts_audio = gr.Audio(label="Listen to it")

        affirmation_btn.click(get_affirmation, outputs=affirmation_output)
        affirmation_output.change(read_aloud, inputs=affirmation_output, outputs=tts_audio)

    with gr.Accordion("🧠 Grounding Techniques", open=False):
        grounding_btn = gr.Button("Give me a grounding technique")
        grounding_output = gr.Textbox(label="Grounding Technique", interactive=False)
        grounding_btn.click(get_grounding_technique, outputs=grounding_output)

demo.launch()



