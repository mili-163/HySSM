import os
import json
import torch
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_omni_utils import process_mm_info


class OmniFeatureExtractor:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"Loading model from {model_path}")
        
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        
        self.SYSTEM_PROMPT = "You are a helpful assistant capable of multi-modal analysis and reasoning."

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        输入文本描述，返回 LLM 的 Hidden State
        策略：将文本作为输入，取最后一层的最后一个 Token 的 embedding
        """
        if not text or len(text.strip()) == 0:
            return torch.zeros(self.model.config.hidden_size, dtype=torch.bfloat16, device=self.device)

        messages = [{"role": "user", "content": text}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        inputs = self.processor(text=[text_input], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            # outputs.hidden_states 是一个 tuple，包含 (layer_0, layer_1, ... layer_N)
            # 取最后一层 [-1]，Batch 第一个 [0]，序列最后一个 Token [-1]
            last_hidden_state = outputs.hidden_states[-1][0, -1, :]
            # Text embedding shape: {last_hidden_state.shape}
            
        return last_hidden_state

    def construct_prompt(self, item):
        """构造 Prompt，保持你原本的逻辑"""
        available_modals = {}
        missing_modals = []
        
        current_visual = item.get('visual', None)
        current_audio = item.get('audio', None)
        item['visual'] = os.path.join("/turing_music_fs/music_data/xyx/zy_workspace/exp/pt/test_20251204/OMG/Code/dataset/process/CH-SIMS", current_visual) if current_visual else None
        item['audio'] = os.path.join("/turing_music_fs/music_data/xyx/zy_workspace/exp/pt/test_20251204/OMG/Code/dataset/process/CH-SIMS", current_audio) if current_audio else None
        # Visual/Audio paths: {item['visual']}, {item['audio']}

        if item['visual'] is None:
            print(f"Missing visual path for item {item['key']}")
            exit(0)
        if item['audio'] is None:
            print(f"Missing audio path for item {item['key']}")
            exit(0)

        if item.get('text'): available_modals['text'] = item['text']
        else: missing_modals.append('text')
        
        # 简单检查文件是否存在
        if item.get('visual') and os.path.exists(item['visual']): available_modals['visual'] = item['visual']
        elif item.get('visual'): missing_modals.append('visual (file missing)') # 路径存在但文件不在
        else: missing_modals.append('visual')
            
        if item.get('audio') and os.path.exists(item['audio']): available_modals['audio'] = item['audio']
        elif item.get('audio'): missing_modals.append('audio (file missing)')
        else: missing_modals.append('audio')

        content = []
        prompt_parts = []
        
        if 'text' in available_modals:
            prompt_parts.append("the given text")
        
        if 'visual' in available_modals:
            if available_modals['visual'].endswith('.mp4'):
                content.append({"type": "video", "video": available_modals['visual']})
                prompt_parts.append("the given video")
            else:
                content.append({"type": "image", "image": available_modals['visual']})
                prompt_parts.append("the given image")

        if 'audio' in available_modals:
            content.append({"type": "audio", "audio": available_modals['audio']})
            prompt_parts.append("the given audio")

        available_prompt = " and ".join(prompt_parts) if prompt_parts else "nothing"
        
        # 如果所有模态都缺失，提供一个 Dummy Prompt 防止报错
        if not prompt_parts and not item.get('text'):
             return None

        if missing_modals:
            missing_modal_name = ','.join(missing_modals)
            task_description = f"predict the features for the missing '{missing_modal_name}' modality"
            missing_instruction = f"STEP 4: Predict decisive feature tags for missing '{missing_modal_name}'."
        else:
            task_description = "analyze all available modalities"
            missing_instruction = "STEP 4: Provide comprehensive feature tags for all modalities."

        prompt_text = f"""
        Based on {available_prompt}, {task_description}.
        
        Please follow this Chain-of-Thought process and then provide the final answer in the specified format.

        <thinking>
        STEP 1 - PLANNING: Identify the available modalities.
        STEP 2 - OVERVIEW: Briefly summarize the overall content.
        STEP 3 - FINE-GRAINED REASONING: Analyze key components.
        {missing_instruction}
        </thinking>

        After your thinking process, provide your answer ONLY in the following JSON format within an <answer> block.

        <answer>
        {{
            "modal_info": {{
                "text": "{'text fine-grained feature long description, detailed'}",
                "visual": "{'visual fine-grained feature long description, illustrate important objects/scenes you see.'}",
                "audio": "{'audio fine-grained feature long description, you need to ASR transcribe speech you hear.'}"
            }}
        }}
        </answer>
        """
        
        content.append({"type": "text", "text": prompt_text})
        

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": content}
        ]
        return conversation

    def process_item(self, item):
        conversation = self.construct_prompt(item)
        if conversation is None:
            print(f"Skipping {item['key']}: missing modalities")
            return None

        try:
            # 1. 准备输入
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audio, images, videos = process_mm_info(conversation, use_audio_in_video=False) 
            
            # 构建 inputs args
            inputs_args = {"text": text, "return_tensors": "pt", "padding": True}
            if images: inputs_args["images"] = images
            if videos: inputs_args["videos"] = videos
            if audio: inputs_args["audio"] = audio
            
            inputs = self.processor(**inputs_args).to(self.device)

            # 2. 生成描述 (Inference)
            with torch.no_grad():
                eos_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048, eos_token_id=eos_token_id)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                # Model output: {output_text}

            # 3. 解析 JSON
            modal_info = {}
            json_match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', output_text, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1))
                    modal_info = parsed_json.get("modal_info", {})
                    # Modal info extracted
                except json.JSONDecodeError:
                    print("JSON parsing failed")
                    pass
            
            # Fallback: 如果没有解析到 audio 描述，但原数据有 audio，则用原文本占位，或者留空
            # 这里的逻辑是：我们希望得到 LLM 对 audio 的"理解"（文本形式）
            t_desc = modal_info.get("text", item.get('text', ""))
            a_desc = modal_info.get("audio", "") # 如果模型没生成，就为空
            v_desc = modal_info.get("visual", "")

            # 4. 提取 Hidden States (Re-encoding)
            # 分别对三个模态的描述进行编码
            t_hidden = self.get_text_embedding(t_desc)
            a_hidden = self.get_text_embedding(a_desc)
            v_hidden = self.get_text_embedding(v_desc)
            
            # 5. 返回结果字典
            return {
                # "key": item['key'],
                "text": t_hidden.cpu(),   # 移回 CPU 节省显存
                "audio": a_hidden.cpu(),
                "visual": v_hidden.cpu(),
                "labels": float(item.get('label', 0.0)),
                "annotation": item.get('annotation', "")
            }

        except Exception as e:
            print(f"Processing error: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/turing_music_fs/music_data/xyx/zy_workspace/exp/pt/test_20251204/OMG/Code/dataset/MOSI/test.jsonl", help="Path to input .jsonl file")
    parser.add_argument("--save_path", type=str, default="/turing_music_fs/music_data/xyx/zy_workspace/exp/pt/test_20251204/OMG/Code/dataset/MOSI/test.pt", help="Path to save output .pt file")
    parser.add_argument("--llm_path", type=str, default="/turing_music_fs/music_data/xyx/zy_workspace/model/Qwen2.5Omni/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00", help="Path to Qwen-Omni model")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 初始化提取器
    extractor = OmniFeatureExtractor(args.llm_path, args.device)

    # 2. 读取数据
    print(f"Loading data from {args.jsonl_path}")
    data_items = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        # for line in f:
        #     if line.strip():
        #         data_items.append(json.loads(line))
        # now is json
        data_items = json.load(f)
    
    # 3. 处理循环
    final_dict = {}

    print(f"Processing {len(data_items)} items")
    for key, item in tqdm(data_items.items()):
        result = extractor.process_item(item)
        # Processed item {key}
        if result:
            # key = result.pop("key") # 提取 key 作为字典的键
            final_dict[key] = result
            
        # 定期保存以防崩溃（可选）
        # if len(final_dict) % 100 == 0:
        #     torch.save(final_dict, args.save_path)

    # 4. 最终保存
    print(f"Saving {len(final_dict)} features to {args.save_path}")
    torch.save(final_dict, args.save_path)

if __name__ == "__main__":
    main()