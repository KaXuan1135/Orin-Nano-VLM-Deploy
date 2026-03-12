# test.py
import numpy as np
import sys
import os

build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cpp/build/bindings'))
sys.path.append(build_path)

import my_engine_binding as engine

def run_test():
    try:
        # 1. 初始化引擎
        print("Initializing engine...")
        # 1. 创建配置对象 (ModelConfig)
        config = engine.ModelConfig()
        
        # 2. 设置必要的参数 (对应你在 C++ 结构体里定义的字段)
        # 假设你的结构体里有这些字段，如果名字不对，请根据实际情况调整
        config.llm_engine_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/InternVL3-1B_llm_engine"
        config.vis_engine_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/InternVL3-1B_vis_engine/model.engine"
        config.tokenizer_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/tokenizers/tokenizer.json"
        config.max_vis_batch = 6

        # 3. 用 config 对象初始化 Runner
        runner = engine.InternVL3Runner(config) # <--- 这里现在就是正确的调用了！
            # 2. 模拟一张 64x64 的 RGB 图片 (H, W, C)
        dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # 构造 input: vector<vector<cv::Mat>> -> list of lists of numpy
        # batch_size = 1, images_per_batch = 1
        images = [[dummy_img]] 
        prompts = ["Describe this image"]
        configs = [engine.GenerateConfig()]
        
        # 3. 调用 C++ 接口
        print("Calling batch_generate...")
        results = runner.batch_generate(images, prompts, configs)
        
        # 4. 输出结果
        for res in results:
            print(f"Output: {res.output_text}")
            
    except Exception as e:
        print(f"Testing failed! Error: {e}")

if __name__ == "__main__":
    run_test()