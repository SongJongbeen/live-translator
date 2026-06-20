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

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY is missing! Please check your .env file.")

transcripts = {
    "source": "",
    "translated": ""
}
ws_app = None

# ---------------------------------------------------------
# WebSocket Event Handlers (표준 gpt-realtime-2 이벤트로 변경)
# ---------------------------------------------------------
def on_message(ws, message):
    try:
        event = json.loads(message)
        event_type = event.get("type")
        
        if event_type == "error":
            logger.error(f"OpenAI API Error: {json.dumps(event.get('error', {}), indent=2)}")
            
        # 번역된 텍스트 수신 (어시스턴트 응답)
        elif event_type == "response.output_audio_transcript.delta":
            delta = event.get("delta", "")
            transcripts["translated"] += delta
            print(delta, end="", flush=True)
            
        # 턴이 끝났을 때 줄바꿈 추가
        elif event_type == "response.done":
            transcripts["translated"] += "\n"
            print() 
            
        # 사용자의 원본 한국어 음성 인식 완료 시점
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
            "type": "realtime", # <-- [필수 추가] GA 정식 버전 필수 파라미터
            "instructions": "You are a highly skilled professional interpreter. Translate the user's spoken Korean into English seamlessly. Only output the translated English text, do not add conversational filler.",
            "audio": {
                "output": {
                    "voice": "alloy" # <-- [위치 변경] 오디오 출력 설정
                },
                "input": {
                    "transcription": {
                        "model": "gpt-realtime-whisper" # <-- [위치 변경] 오디오 입력 설정
                    }
                }
            }
        }
    }
    ws.send(json.dumps(config))

def start_websocket():
    global ws_app
    # 전용 번역 엔드포인트가 아닌 일반 Realtime 엔드포인트 사용
    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"
    headers = [f"Authorization: Bearer {os.getenv('OPENAI_API_KEY')}"]
    
    ws_app = websocket.WebSocketApp(
        url, header=headers,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close
    )
    ws_app.run_forever()

threading.Thread(target=start_websocket, daemon=True).start()

# ---------------------------------------------------------
# PDF Upload Handler (동적 세션 업데이트)
# ---------------------------------------------------------
def on_pdf_upload(file_info):
    global ws_app
    if not file_info:
        return "파일이 업로드되지 않았습니다."
    if ws_app is None or not ws_app.sock or not ws_app.sock.connected:
        return "WebSocket이 아직 연결되지 않았습니다. 잠시 후 다시 시도하세요."
    
    try:
        reader = PyPDF2.PdfReader(file_info.name)
        pdf_text = "".join([page.extract_text() + "\n" for page in reader.pages])
        
        # 모델 컨텍스트 윈도우 보호를 위해 너무 큰 파일은 자릅니다 (약 5000자)
        pdf_text = pdf_text[:5000] 
        
        new_instructions = f"You are a highly skilled professional interpreter. Translate the user's spoken Korean into English seamlessly (or if the user speaks in English, translate it seamlessly into Korean). Use the following reference document to understand the context and utilize specific terminology:\n\n<REFERENCE_DOCUMENT>\n{pdf_text}\n</REFERENCE_DOCUMENT>\n\nOnly output the translated English text."
        
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
        
        # 번역 세션 전용이 아니므로 'session.' 접두사를 제거합니다.
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
    
    with gr.Row():
        pdf_upload = gr.File(label="참고용 PDF 업로드 (선택사항)", file_types=[".pdf"])
        status_text = gr.Textbox(label="시스템 상태", interactive=False)
        
    with gr.Row():
        source_box = gr.Textbox(label="Transcribed", lines=10, interactive=False)
        translated_box = gr.Textbox(label="Translated", lines=10, interactive=False)
            
    mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
    
    pdf_upload.upload(fn=on_pdf_upload, inputs=[pdf_upload], outputs=[status_text])
    mic.stream(fn=process_audio, inputs=[mic], outputs=[source_box, translated_box])

if __name__ == "__main__":
    demo.launch()
