import time
import numpy as np
import cv2

class InferenceManager:
    def __init__(self, engine_module):
        self.engine = engine_module
        config = self.engine.ModelConfig()
        config.llm_engine_path = "/mnt/sdcard/models/InternVL3-1B_i8/InternVL3-1B_llm_engine"
        config.vis_engine_path = "/mnt/sdcard/models/InternVL3-1B_i8/InternVL3-1B_vis_engine/model.engine"
        config.tokenizer_path = "/mnt/sdcard/models/InternVL3-1B_i8/tokenizers/tokenizer.json"
        config.max_vis_batch = 6
        self.runner = self.engine.AsyncInternVL3Runner(config)
        self.previous_handles = {}

    def stream_infer(self, message, image_data, session_id):
        """
        image_data: Gradio 传来的通常是一个字典 {'image': ..., 'text': ...} 
                    或者是图片文件路径，或者是 PIL Image 对象
        """
        if image_data is not None:
            img_np = np.array(image_data.convert('RGB'))
            img_mat = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            images = [img_mat] 
        else:
            images = []

        gen_config = self.engine.GenerateConfig()
        gen_config.system_prompt = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫InternVL, 是一个有用无害的人工智能助手。"
        gen_config.streaming = True
        gen_config.profiling = True

        if session_id not in self.previous_handles.keys():
            self.previous_handles[session_id] = []

        handle = self.runner.enqueue_chat([images], [message], [gen_config], self.previous_handles[session_id])
        if session_id in self.previous_handles.keys():
            self.previous_handles[session_id].append(handle)
        else:
            self.previous_handles[session_id] = [handle]
        
        full_text = ""
        while not handle.gen_finished:
            new_tokens = handle.pop_last_outputs_text()
            if new_tokens and len(new_tokens) > 0:
                token = new_tokens[0]
                full_text += token
                yield full_text # Gradio 的流式魔法
            time.sleep(0.01)
            
        final_tokens = handle.pop_last_outputs_text()
        if final_tokens:
            yield full_text + final_tokens[0]

        stats = [
            ["Latency", f"{handle.generation_latency_ms:.2f} ms"],
            ["TTFT", f"{handle.time_to_first_token_ms:.2f} ms"],
            ["Throughput", f"{handle.tokens_per_second:.2f} tokens/s"],
            ["Input Tokens", f"{handle.input_tokens_len}"],
            ["Output Tokens", f"{handle.output_tokens_len}"]
        ]

        print("\n" + "="*30)
        print(f"{'VLM Inference Metrics':^30}")
        print("-"*30)
        for label, value in stats:
            print(f"{label:<15} : {value:>12}")
        print("="*30 + "\n")
# 全局单例管理器
