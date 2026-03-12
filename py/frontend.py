import gradio as gr
# 假设这是你封装好的 C++ 推理引擎模块
# import your_cpp_binding_module 

# engine = your_cpp_binding_module.InferenceEngine()


def chat_stream(message, history, session_id):
    # 这里接入你的 C++ 接口
    # 你的 C++ 引擎应该能在内部处理不同 session_id 的请求
    partial_text = ""
    for token in engine.infer_stream(message, session_id):
        partial_text += token
        yield partial_text

with gr.Blocks(title="In-flight Batching Demo") as demo:
    gr.Markdown("# EdgeVLM: Concurrency & In-flight Batching Demo")
    
    with gr.Row():
        # 左侧窗口
        with gr.Column():
            chat1 = gr.Chatbot(label="Session 1 (Batch 1)")
            msg1 = gr.Textbox(label="Input 1")
            msg1.submit(chat_stream, [msg1, chat1, gr.State("session_1")], [chat1])
            
        # 右侧窗口
        with gr.Column():
            chat2 = gr.Chatbot(label="Session 2 (Batch 2)")
            msg2 = gr.Textbox(label="Input 2")
            msg2.submit(chat_stream, [msg2, chat2, gr.State("session_2")], [chat2])

demo.queue().launch()