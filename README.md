# EXAONE-4.0-32B Chatbot (HuggingFace 포맷)
> Hugging Face Transformers + Gradio로 만든 대화형 LLM
------------------------

#### 이 프로젝트는 EXAONE-4.0-32B 모델을 Hugging Face transformers 라이브러리를 사용해 불러오고,
#### 양자화(8bit/4bit) 를 적용하여 GPU 메모리를 효율적으로 활용하면서 대화할 수 있도록 구현된 예제입니다.

## ✅ Requirements
---------------------
OS: ![Ubuntu](https://img.shields.io/badge/Ubuntu-18.04-E95420?logo=ubuntu&logoColor=white)

GPU: ![NVIDIA](https://img.shields.io/badge/NVIDIA-A100-76B900?logo=nvidia&logoColor=white)

Python: ![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white) (Anaconda 권장)

CUDA: ![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)

glibc: ![glibc](https://img.shields.io/badge/glibc-2.29-blue)

## 🚀 Features
---------------
대형 언어모델(EXAONE 32B) 지원

양자화(8bit/4bit) 옵션으로 GPU 메모리 절약

대화 맥락 유지 (Chat Template 기반)

웹 UI (Gradio) 제공 → 실시간 채팅 가능

생성 파라미터 조절

temperature: 창의성 조절

top_p: 확률적 샘플링

max_new_tokens: 답변 최대 길이 설정

## 📦 Installation
-------------------
##### 1. Conda 환경 생성
conda create -n exaone python=3.10 -y
conda activate exaone

##### 2. PyTorch 설치 (CUDA 12.1)
pip install torch==2.1.2+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

##### 3. Transformers, Accelerate, BitsAndBytes 설치
pip install transformers==4.44.0 accelerate==0.34.0 bitsandbytes gradio

##### 4. 모델 다운로드
git lfs install
git clone https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B ./exaone_model

##### 5. bitsandbytes gradio 패키지 설치
pip install torch transformers bitsandbytes gradio


⚠️ GPU 환경 (예: A100)에서 실행 권장

#### ▶️ Usage
##### 1. 서버 실행
    python chat_exaone.py

##### 2. 웹 브라우저 접속
    http://localhost:7860

##### 💻 Code Overview
###### 모델 불러오기
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "LGAI-EXAONE/EXAONE-4.0-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

###### 채팅 함수
    def chat_fn(message, history):
    chat_history.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
       **inputs,
       max_new_tokens=512,
       do_sample=True,
       temperature=0.7,
       top_p=0.9
    )

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    chat_history.append({"role": "assistant", "content": reply})
    return reply

    Gradio UI
    import gradio as gr

    demo = gr.ChatInterface(fn=chat_fn)
    demo.launch(server_name="0.0.0.0", server_port=7860)

## ⚖️ Notes
---------
###### 양자화(Quantization)

load_in_8bit=True: 메모리 절약 + 속도 향상

load_in_4bit=True: VRAM이 부족할 경우 사용 (정밀도 약간 손실 가능)

GPU 메모리 사용량

FP16: 매우 큼 (32B 모델은 단일 GPU 불가)

8bit: 절반 수준

4bit: 1/4 수준

## 📜 License
-------------
본 프로젝트는 연구/학습 목적의 예제 코드입니다.
실제 서비스에 사용할 경우 모델 라이선스를 반드시 확인하세요.
