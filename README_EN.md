# KoBART Question Generation

This repository contains an example notebook that demonstrates how to **automatically generate questions** from Korean contexts using the **KoBART model**.  
It can be applied to research, education, dataset creation, and more.

---

## 🚀 Features
- Question generation based on KoBART (using Hugging Face Transformers)
- Generate multiple questions at once (`generate_multiple_questions` function)
- Supports parameters such as Beam Search and no-repeat n-gram
- Test results with simple example contexts
- ⚠️ Currently works only with **Korean input**

---

## 📂 Repository Structure
```
.
├── kobart_question_generation_fixed.ipynb
└── README_EN.md
```
> `kobart_question_generation_fixed.ipynb` is a cleaned version with `metadata.widgets` JSON fixed.

---

## ⚙️ Installation

Python 3.8+ is recommended.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install torch transformers
```
> ⚠️ If CUDA is available, GPU can be used (automatically detected by `torch.cuda.is_available()`).

---

## 🖥️ Usage (Notebook)

Run the notebook in Jupyter Notebook or Google Colab.

### 1) Load Model & Tokenizer
```python
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v1")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

### 2) Question Generation Function
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
            num_beams=10,  # number of beams (higher → more diverse)
            num_return_sequences=num_questions,  # generate multiple questions
            early_stopping=True,
            no_repeat_ngram_size=2  # prevent repetition
        )

        print(f"=== Context {i+1} ===")
        print(f"텍스트: {context}\n")
        for idx, output in enumerate(outputs):
            question = tokenizer.decode(output, skip_special_tokens=True)
            print(f"생성된 질문 {idx+1}: {question}")
        print("\n")
```

### 3) Run with Example Contexts
```python
contexts = [
    "지구는 태양 주위를 공전하면서 사계절의 변화를 만들어낸다. 이 과정에서 지구의 자전축이 기울어져 있기 때문에 각 지역은 계절마다 다른 기온과 기후를 경험하게 된다.",
    "산업 혁명은 18세기 후반 영국에서 시작되어 전 세계로 확산되었다. 증기기관과 기계화된 생산 방식은 사회 구조와 경제 체제를 근본적으로 바꾸어 놓았다."
]

generate_multiple_questions(model, tokenizer, contexts, num_questions=3)
```

---

## 📝 Example Output
Input Context:
```
달의 중력은 지구의 바닷물에 영향을 주어 밀물과 썰물이 규칙적으로 발생한다.
```

Generated Questions (in Korean):
```
1. 달의 중력은 지구에 어떤 영향을 미치는가?
2. 밀물과 썰물이 발생하는 원인은 무엇인가?
3. 조석 현상은 어떻게 설명할 수 있는가?
```

---

## ⚠️ Notes
- A higher `num_beams` increases diversity but can slow down generation (especially in Colab).
- Question quality may vary depending on context length, sentence structure, and domain.
- To avoid syntax errors, ensure parentheses and comments in the `model.generate(` block are properly separated.
- This model is trained on Korean and therefore works **only with Korean input**.

---

## 🤝 Contributing
Contributions and PRs are welcome. For bug reports, please include reproducible code snippets and environment details.

---

## 📜 License
MIT License
