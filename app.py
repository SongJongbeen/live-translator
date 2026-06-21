import os
import json
import base64
import threading
import logging
import numpy as np
import gradio as gr
import websocket
import librosa
import PyPDF2
from dotenv import load_dotenv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

transcripts = {
    "source": "",
    "translated": ""
}
ws_app = None
ws_thread = None

# ---------------------------------------------------------
# WebSocket Event Handlers
# ---------------------------------------------------------
def on_message(ws, message):
    try:
        event = json.loads(message)
        event_type = event.get("type")
        
        if event_type == "error":
            logger.error(f"OpenAI API Error: {json.dumps(event.get('error', {}), indent=2)}")
            
        elif event_type == "response.output_audio_transcript.delta":
            delta = event.get("delta", "")
            transcripts["translated"] += delta
            print(delta, end="", flush=True)
            
        elif event_type == "response.done":
            transcripts["translated"] += "\n"
            print() 
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcripts["source"] += event.get("transcript", "") + "\n\n"
            
    except Exception as e:
        logger.error(f"Error parsing message: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket Network Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning("WebSocket closed.")

def on_open(ws):
    logger.info("WebSocket connected to OpenAI Standard Realtime API.")
    
    config = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "You are a highly skilled professional interpreter for an academic conference. Translate the user's spoken Korean into English seamlessly. Or if the user speaks English, then translate it into Korean.",
            "audio": {
                "output": {
                    "voice": "alloy" 
                },
                "input": {
                    "transcription": {
                        "model": "gpt-realtime-whisper" 
                    }
                }
            }
        }
    }
    ws.send(json.dumps(config))

def start_websocket():
    global ws_app
    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"
    headers = [f"Authorization: Bearer {OPENAI_API_KEY}"]
    
    ws_app = websocket.WebSocketApp(
        url, header=headers,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close
    )
    ws_app.run_forever()

# ---------------------------------------------------------
# Connection Management (새로 추가된 부분)
# ---------------------------------------------------------
def connect_server():
    global ws_thread, ws_app
    if ws_app and ws_app.sock and ws_app.sock.connected:
        return "🟢 이미 서버에 연결되어 있습니다."
    
    ws_thread = threading.Thread(target=start_websocket, daemon=True)
    ws_thread.start()
    return "🟡 서버 연결 중... (음성 입력을 시작하셔도 됩니다)"

def disconnect_server():
    global ws_app
    if ws_app:
        ws_app.close()
        return "🔴 서버 연결이 종료되었습니다."
    return "서버가 이미 종료되어 있습니다."

# ---------------------------------------------------------
# PDF Upload Handler
# ---------------------------------------------------------
def on_pdf_upload(file_info):
    global ws_app
    if not file_info:
        return "파일이 업로드되지 않았습니다."
    if ws_app is None or not ws_app.sock or not ws_app.sock.connected:
        return "⚠️ WebSocket이 연결되지 않았습니다. 먼저 '서버 연결' 버튼을 눌러주세요."
    
    try:
        reader = PyPDF2.PdfReader(file_info.name)
        pdf_text = "".join([page.extract_text() + "\n" for page in reader.pages])
        pdf_text = pdf_text[:5000] 
        
        new_instructions = f"You are a highly skilled professional interpreter for an academic conference. Translate the user's spoken Korean into English seamlessly (or if the user speaks in English, translate it seamlessly into Korean). Use the following reference document to understand the context and utilize specific terminology:\n\n<REFERENCE_DOCUMENT>\n{pdf_text}\n</REFERENCE_DOCUMENT>"
        
        config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": new_instructions
            }
        }
        ws_app.send(json.dumps(config))
        return "✅ PDF 분석 완료! 이제 문서의 문맥을 반영하여 번역합니다."
        
    except Exception as e:
        return f"❌ PDF 처리 오류: {str(e)}"

# ---------------------------------------------------------
# Gradio Audio Processing
# ---------------------------------------------------------
def process_audio(audio_chunk):
    global ws_app
    
    if audio_chunk is None or ws_app is None or not ws_app.sock or not ws_app.sock.connected:
        return transcripts["source"], transcripts["translated"]
    
    sample_rate, audio_data = audio_chunk
    
    try:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
            
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        if sample_rate != 24000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=24000)
        
        pcm16 = np.int16(audio_data * 32767).tobytes()
        base64_audio = base64.b64encode(pcm16).decode("utf-8")
        
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        ws_app.send(json.dumps(event))
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
    
    return transcripts["source"], transcripts["translated"]

# ---------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------
with gr.Blocks(title="고려대학교 동시통역 프로그램") as demo:
    gr.Markdown("# 고려대학교 동시통역 프로그램")
    gr.Markdown("문의: 송종빈 (1041489@gmail.com)")
    
    # 서버 연결 제어 버튼 (추가됨)
    with gr.Row():
        btn_connect = gr.Button("🟢 서버 연결 시작", variant="primary")
        btn_disconnect = gr.Button("🔴 서버 연결 종료", variant="stop")
        ws_status_text = gr.Textbox(label="서버 연결 상태", value="🔴 연결되지 않음 (사용 전 연결 버튼을 누르세요)", interactive=False)
        
    with gr.Row():
        pdf_upload = gr.File(label="참고용 PDF 업로드 (선택사항)", file_types=[".pdf"])
        status_text = gr.Textbox(label="시스템 상태", interactive=False)
        
    with gr.Row():
        source_box = gr.Textbox(label="Transcribed", lines=10, interactive=False)
        translated_box = gr.Textbox(label="Translated", lines=10, interactive=False)
            
    mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
    
    # Event Listeners
    btn_connect.click(fn=connect_server, inputs=[], outputs=[ws_status_text])
    btn_disconnect.click(fn=disconnect_server, inputs=[], outputs=[ws_status_text])
    
    pdf_upload.upload(fn=on_pdf_upload, inputs=[pdf_upload], outputs=[status_text])
    mic.stream(fn=process_audio, inputs=[mic], outputs=[source_box, translated_box])

if __name__ == "__main__":
    # 기존에 백그라운드에서 무조건 웹소켓을 실행하던 코드는 삭제했습니다.
    # 로컬에서 포트로 실행됨
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080))
    )
