import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests
from gtts import gTTS
import whisper
from deep_translator import GoogleTranslator
from datetime import datetime

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global state
chunks = []
sources = []
index = None
chat_history = []
all_files = []

def extract_text_with_page(pdf_files):
    all_chunks = []
    all_sources = []
    for file in pdf_files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                split_chunks = [text[j:j+500] for j in range(0, len(text), 500)]
                all_chunks.extend(split_chunks)
                all_sources.extend([f"{file.name} - Page {i+1}"] * len(split_chunks))
    print(f"✅ Extracted {len(all_chunks)} chunks from {len(pdf_files)} PDF(s).")
    return all_chunks, all_sources

def build_vector_store(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search(query, chunks, sources, index, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [(chunks[i], sources[i]) for i in I[0]]

def ask_groq(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[Groq API Error] No API key found."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Groq API Error] {e}"

def text_to_speech(text, filename="answer.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

def handle_upload(files):
    global chunks, sources, index, chat_history, all_files
    all_files = files
    chunks, sources = extract_text_with_page(files)
    if not chunks:
        return ["All PDFs"], "⚠️ No readable text found in uploaded PDFs.", "All PDFs"
    index = build_vector_store(chunks)
    chat_history = []
    file_names = [f.name for f in files]
    default_file = file_names[0] if len(files) == 1 else "All PDFs"
    return ["All PDFs"] + file_names, f"✅ Processed {len(files)} PDF(s). Ready to answer questions.", default_file

def handle_question(user_input, lang_choice, selected_file):
    global chat_history, index
    if not index or not chunks:
        return [("⚠️ Please upload and process PDF(s) before asking questions.", "")], None

    if lang_choice == "Urdu":
        translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
    else:
        translated_input = user_input

    try:
        if selected_file != "All PDFs":
            filtered_chunks = [c for c, s in zip(chunks, sources) if s.startswith(selected_file)]
            filtered_sources = [s for s in sources if s.startswith(selected_file)]
            print(f"📂 Filtering chunks for: {selected_file} → {len(filtered_chunks)} chunks")
            if not filtered_chunks:
                return chat_history + [("⚠️ No content found in selected file.", "")], None
            local_index = build_vector_store(filtered_chunks)
            top_chunks = search(translated_input, filtered_chunks, filtered_sources, local_index)
        else:
            top_chunks = search(translated_input, chunks, sources, index)

        context = "\n".join([f"[{src}] {txt}" for txt, src in top_chunks])
        prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {translated_input}"
        answer = ask_groq(prompt)

        if lang_choice == "Urdu":
            answer = GoogleTranslator(source='auto', target='ur').translate(answer)

        chat_history.append((user_input, answer))
        audio_path = text_to_speech(answer)
        return chat_history, audio_path
    except Exception as e:
        return chat_history + [("⚠️ Error during question processing.", str(e))], None

def handle_voice(audio_path, lang_choice, selected_file):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    transcript = result["text"]
    return handle_question(transcript, lang_choice, selected_file)

def clear_all():
    global chat_history
    chat_history = []
    return [], None

def download_chat():
    global chat_history
    if not chat_history:
        return None, "⚠️ No chat history to export."
    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(os.getcwd(), filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for q, a in chat_history:
                f.write(f"You: {q}\nBot: {a}\n\n")
        return path, f"✅ Chat history exported to {filename}"
    except Exception as e:
        return None, f"⚠️ Error exporting chat history: {str(e)}"

css = '''
body {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(to bottom, #f1f5f9, #e2e8f0);
    color: #1e293b;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}
#navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to right, #14b8a6, #06b6d4);
    color: #ffffff;
    padding: 12px 24px;
    font-size: 20px;
    font-weight: 600;
    z-index: 1000;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
}
#navbar::before {
    content: "📘";
    margin-right: 8px;
}
#title {
    font-size: 28px;
    font-weight: 700;
    color: #14b8a6;
    margin-bottom: 16px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.gr-box {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 24px;
    margin-bottom: 20px;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.gr-box:hover {
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}
.gr-button {
    font-weight: 600;
    border-radius: 8px;
    background: #14b8a6;
    color: #ffffff;
    padding: 10px 16px;
    border: none;
    transition: background 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
}
.gr-button:hover {
    background: #0d9488;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.gr-textbox, .gr-dropdown, .gr-file {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.gr-textbox:focus, .gr-dropdown:focus, .gr-file:focus {
    border-color: #14b8a6;
    box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.2);
}
.gr-chatbot .message {
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
#about-btn {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #8b5cf6;
    color: #ffffff;
    padding: 12px 20px;
    border-radius: 50px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    z-index: 1000;
    transition: background 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
}
#about-btn:hover {
    background: #7c3aed;
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
footer {
    text-align: center;
    font-size: 14px;
    color: #4b5563;
    margin: 24px auto;
    padding: 16px;
    max-width: 800px;
    width: 100%;
}
footer a {
    color: #14b8a6;
    text-decoration: none;
    margin: 0 8px;
}
footer a:hover {
    text-decoration: underline;
    color: #0d9488;
}
'''

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.HTML('<div id="navbar">Smart PDF ChatBot - by Noman Shamim</div>')
    gr.Markdown("## 🤖 <span id='title'>Smart PDF ChatBot Assistant</span>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload PDF")
            file_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF(s)")
            upload_btn = gr.Button("Process PDF")
            file_list = gr.Dropdown(choices=["All PDFs"], label="Select File", value="All PDFs", allow_custom_value=True)
            status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### 🌐 Language")
            lang_dropdown = gr.Dropdown(choices=["English", "Urdu"], value="English", label="Language")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask or Speak Your Question")
            chatbot = gr.Chatbot(label="Chat History")
            with gr.Row():
                mic_input = gr.Microphone(type="filepath", label="🎤 Speak")
                txt_input = gr.Textbox(label="Type your question")
            with gr.Row():
                send_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")
                export_btn = gr.Button("📄 Export Chat")
            audio_output = gr.Audio(label="🔊 Voice Answer")
            export_status = gr.Textbox(label="Export Status", interactive=False)

    upload_btn.click(
        fn=handle_upload,
        inputs=[file_input],
        outputs=[file_list, status, file_list]
    )
    send_btn.click(
        fn=handle_question,
        inputs=[txt_input, lang_dropdown, file_list],
        outputs=[chatbot, audio_output]
    )
    mic_input.change(
        fn=handle_voice,
        inputs=[mic_input, lang_dropdown, file_list],
        outputs=[chatbot, audio_output]
    )
    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, audio_output]
    )
    export_btn.click(
        fn=download_chat,
        outputs=[gr.File(label="Download Chat"), export_status]
    )

    gr.Markdown("""
    <footer>
        Made by <strong>Noman Shamim</strong> ❤️ using <strong>LLaMA3</strong>, <strong>Gradio</strong>, and <strong>FAISS</strong>.<br>
        <a href="https://github.com/NomanShamim" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/noman-shamim-00a0662b1" target="_blank">LinkedIn</a>
    </footer>
    """)

    gr.HTML('''
    <button id="about-btn" onclick="alert('This chatbot allows you to upload PDFs, ask questions by voice or text, and get answers from LLaMA3. Built by Noman Shamim.')">About</button>
    ''')

demo.launch()