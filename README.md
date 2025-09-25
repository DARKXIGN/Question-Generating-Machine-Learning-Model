# KoBART Question Generation

이 저장소는 **KoBART 모델**을 활용하여 한국어 문맥(Context)으로부터 **질문을 자동 생성**하는 예제 노트북을 포함합니다.  
연구, 교육, 데이터셋 생성 등 다양한 목적에서 활용할 수 있습니다.

---

## 🚀 Features
- KoBART 기반 질문 생성 (Hugging Face Transformers 활용)
- 여러 개의 질문을 한 번에 생성 (`generate_multiple_questions` 함수)
- Beam Search, 반복 방지 옵션 등 다양한 파라미터 제공
- 간단한 예시 Context로 결과 확인

---

## 📂 Repository Structure
```
.
├── kobart_question_generation_fixed.ipynb
└── README.md
```
> `kobart_question_generation_fixed.ipynb` 는 `metadata.widgets` json 파일을 수정한 버전입니다.

---

## ⚙️ Installation

Python 3.8+ 환경을 권장합니다.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install torch transformers
```
> ⚠️ CUDA 환경이 있다면 GPU 사용이 가능합니다. (`torch.cuda.is_available()` 자동 감지)

---

## 🖥️ Usage (Notebook)

Jupyter Notebook 또는 Google Colab에서 노트북 파일을 실행하세요.

### 1) 모델 & 토크나이저 로드
```python
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v1")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

### 2) 질문 생성 함수 (수정된 버전 예시)
```python
def generate_multiple_questions(model, tokenizer, contexts, max_length=64, num_questions=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for i, context in enumerate(contexts):
        inputs = tokenizer(
            context,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=10,  # 빔서치 수 (더 높으면 다양성↑)
            num_return_sequences=num_questions,  # 여러 개 질문 생성
            early_stopping=True,
            no_repeat_ngram_size=2  # 반복 방지
        )

        print(f"=== Context {i+1} ===")
        print(f"텍스트: {context}\\n")
        for idx, output in enumerate(outputs):
            question = tokenizer.decode(output, skip_special_tokens=True)
            print(f"생성된 질문 {idx+1}: {question}")
        print("\\n")
```

### 3) 예시 Context로 실행
```python
contexts = [
    "지구는 태양 주위를 공전하면서 사계절의 변화를 만들어낸다. 이 과정에서 지구의 자전축이 기울어져 있기 때문에 각 지역은 계절마다 다른 기온과 기후를 경험하게 된다.",
    "산업 혁명은 18세기 후반 영국에서 시작되어 전 세계로 확산되었다. 증기기관과 기계화된 생산 방식은 사회 구조와 경제 체제를 근본적으로 바꾸어 놓았다."
]

generate_multiple_questions(model, tokenizer, contexts, num_questions=3)
```

---

## 📝 Example Output
입력 Context:
```
달의 중력은 지구의 바닷물에 영향을 주어 밀물과 썰물이 규칙적으로 발생한다.
```

생성된 질문 예시:
```
1. 달의 중력은 지구에 어떤 영향을 미치는가?
2. 밀물과 썰물이 발생하는 원인은 무엇인가?
3. 조석 현상은 어떻게 설명할 수 있는가?
```

---

## ⚠️ Notes
- `num_beams`가 높을수록 다양성이 증가하지만 속도가 느려질 수 있습니다.(특히 코랩 환경에서 매우 느리게 작동합니다...)
- 질문 품질은 Context의 길이·문장 구조·도메인에 따라 달라질 수 있습니다.
- 괄호 오류가 재발하지 않도록 `model.generate(` 블록의 인자/괄호 짝을 주석과 분리해 작성하세요.

---

## 🤝 Contributing
이슈/PR 환영합니다. 버그 리포트 시 재현 가능한 코드 조각과 환경 정보를 포함해 주세요.

---

## 📜 License
MIT License
