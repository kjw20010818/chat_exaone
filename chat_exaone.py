import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 모델 경로
model_id = "LGAI-EXAONE/EXAONE-4.0-32B"

print("🔄 모델 로드 중... (시간이 걸릴 수 있습니다)")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 양자화 설정 (8bit → 4bit로도 변경 가능)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,      # VRAM 여유 없으면 load_in_4bit=True 로 변경
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,   # A100이니 fp16 권장
    quantization_config=bnb_config
)

# 대화 기록 저장
chat_history = []

def chat_fn(message, history):
    global chat_history
    chat_history.append({"role": "user", "content": message})

    # chat_template 적용
    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,   # 답변 길이 늘림
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # 새로 생성된 부분만 추출
    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    chat_history.append({"role": "assistant", "content": reply})
    return reply

demo = gr.ChatInterface(fn=chat_fn)
demo.launch(server_name="0.0.0.0", server_port=7860)
