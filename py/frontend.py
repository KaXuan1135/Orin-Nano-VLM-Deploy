import os
import sys
import gradio as gr

from PIL import Image
from gradio import ChatMessage
from inference_manager import InferenceManager

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../cpp/build/bindings'))
)
import my_engine_binding as engine

manager = InferenceManager(engine)

def stream_chat(message, history, session_id):
    user_text = message.get("text", "")
    files = message.get("files", [])
    
    content_list = []
    if user_text:
        content_list.append({"type": "text", "text": user_text})
    if files:
        content_list.append({"type": "image", "path": files[0]})
        
    user_msg = ChatMessage(role="user", content=content_list)
    assistant_msg = ChatMessage(role="assistant", content="")
    
    history.append(user_msg)
    history.append(assistant_msg)

    # 2. Inference
    image = Image.open(files[0]) if files else None
    
    for partial_text in manager.stream_infer(user_text, image, session_id):
        # 3. Update the content of the last message (assistant)
        history[-1].content = partial_text
        yield history

with gr.Blocks(title="EdgeVLM Dual-Stream Demo") as demo:
    gr.Markdown("# EdgeVLM: Concurrency & System Integration Demo")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Session A (Stream 1)")
            chat1 = gr.Chatbot(label="VLM Inference 1")
            msg1 = gr.MultimodalTextbox(label="Input 1")
            # session_id 作为 State 传入
            state1 = gr.State("session_A") 
            msg1.submit(stream_chat, [msg1, chat1, state1], [chat1])
            
        with gr.Column():
            gr.Markdown("### Session B (Stream 2)")
            chat2 = gr.Chatbot(label="VLM Inference 2")
            msg2 = gr.MultimodalTextbox(label="Input 2")
            state2 = gr.State("session_B")
            msg2.submit(stream_chat, [msg2, chat2, state2], [chat2])

demo.queue().launch()