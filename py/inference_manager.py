import time
import numpy as np
import cv2

class InferenceManager:
    def __init__(self, engine_module):
        self.engine = engine_module
        config = self.engine.ModelConfig()
        config.llm_engine_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/InternVL3-1B_llm_engine"
        config.vis_engine_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/InternVL3-1B_vis_engine/model.engine"
        config.tokenizer_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/tokenizers/tokenizer.json"
        config.max_vis_batch = 6
        self.runner = self.engine.AsyncInternVL3Runner(config)

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
        gen_config.streaming = True

        handle = self.runner.enqueue_generate([images], [message], [gen_config])
        
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

# 全局单例管理器
