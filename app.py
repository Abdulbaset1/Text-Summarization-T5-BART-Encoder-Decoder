import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

st.set_page_config(page_title="T5 LoRA Summarizer", layout="wide")

st.title("📄 T5 Summarizer (LoRA Fine-Tuned)")
st.write("Model: **basit1878/t5-small-lora-summarizer**")

@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, "basit1878/t5-small-lora-summarizer")

    # Merge LoRA into base model for inference
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    return model, tokenizer

model, tokenizer = load_model()

# -------------------------------
# Input Area
# -------------------------------
text = st.text_area("Enter text to summarize:", height=250)

# -------------------------------
# Generation Settings
# -------------------------------
st.subheader("⚙️ Generation Settings")

use_sampling = st.checkbox("Use creative sampling (more abstractive)", value=False)

if use_sampling:
    st.write("Sampling enabled → More paraphrasing, less copying.")

max_tokens = st.slider("Maximum summary tokens", 50, 300, 180)

# -------------------------------
# Summarize Button
# -------------------------------
if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # ✨ Better summarization prompt
        prompt = "Write a concise, original, and non-repetitive summary of the following text:\n\n" + text

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # -------------------------------
        # GENERATION LOGIC
        # -------------------------------
        with torch.no_grad():
            if use_sampling:
                # More creative / abstractive
                outputs = model.generate(
                    **inputs,
                    max_length=max_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.35,
                )
            else:
                # More accurate / controlled
                outputs = model.generate(
                    **inputs,
                    max_length=max_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.4,
                )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -------------------------------
        # Output
        # -------------------------------
        st.subheader("📌 Summary")
        st.write(summary)
