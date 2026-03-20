import os
import sys
import nvtx
import asyncio
import gradio as gr

from PIL import Image
from gradio import ChatMessage
from gradio_backend import InferenceManager

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../cpp/build/bindings'))
)
import my_engine_binding as engine

manager = InferenceManager(engine)

async def stream_chat(message, history, session_id):
    with nvtx.annotate("FastAPI_Request_Process", color="green"):
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

        with nvtx.annotate("Engine_Setup_and_Infer", color="red"):
            image = Image.open(files[0]) if files else None
            
            loop = asyncio.get_event_loop()
            
            def get_iter():
                return manager.stream_infer(user_text, image, session_id)

            it = await loop.run_in_executor(None, get_iter)

            while True:
                def get_next(i):
                    try:
                        return next(i)
                    except StopIteration:
                        return None

                partial_text = await loop.run_in_executor(None, get_next, it)
                if partial_text is None:
                    break

                with nvtx.annotate("Update_History"):
                    history[-1].content = partial_text
                    yield history
                
                await asyncio.sleep(0.01)


with gr.Blocks(title="EdgeVLM Dual-Stream Demo") as demo:
    gr.Markdown("# EdgeVLM: Concurrency & System Integration Demo")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Session A (Stream 1)")
            chat1 = gr.Chatbot(label="VLM Inference 1")
            msg1 = gr.MultimodalTextbox(label="Input 1")
            state1 = gr.State("session_A") 
            msg1.submit(stream_chat, [msg1, chat1, state1], [chat1])
            
        with gr.Column():
            gr.Markdown("### Session B (Stream 2)")
            chat2 = gr.Chatbot(label="VLM Inference 2")
            msg2 = gr.MultimodalTextbox(label="Input 2")
            state2 = gr.State("session_B")
            msg2.submit(stream_chat, [msg2, chat2, state2], [chat2])

    demo.load(fn=manager.reset_history, inputs=None, outputs=None)

demo.queue(default_concurrency_limit=2).launch()