import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

st.set_page_config(page_title="T5 LoRA Summarizer", layout="wide")

st.title("📄 T5 Summarizer (LoRA Fine-Tuned)")
st.write("Model: **basit1878/t5-small-lora-summarizer**")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, "basit1878/t5-small-lora-summarizer")

    # Merge LoRA into base model for faster inference
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    return model, tokenizer

model, tokenizer = load_model()


# -------------------------------
# Input Text
# -------------------------------
text = st.text_area("Enter text to summarize:", height=260)


# -------------------------------
# Settings Panel
# -------------------------------
st.subheader("⚙️ Generation Settings")

use_sampling = st.radio(
    "Generation Mode",
    ["Accurate (Beam Search)", "Creative (Sampling)"],
    horizontal=True
)

max_tokens = st.slider("Maximum summary length (tokens)", 50, 300, 180)


# -------------------------------
# Summarization Logic
# -------------------------------
if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:

        # Better prompt → forces abstractive summaries
        prompt = (
            "Write a clear, concise, fluent, and original summary of the following text. "
            "Do NOT copy exact sentences. Rewrite it in your own words:\n\n"
            + text
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # -------------------------------
        # Beam Search (More Accurate)
        # -------------------------------
        if use_sampling == "Accurate (Beam Search)":
            outputs = model.generate(
                **inputs,
                max_length=max_tokens,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=1.55,   # Stronger anti-copying
                min_length=60,              # Ensures complete sentences
                length_penalty=1.2,         # Encourages good structure
            )

        # -------------------------------
        # Sampling (More Abstractive)
        # -------------------------------
        else:
            outputs = model.generate(
                **inputs,
                max_length=max_tokens,
                do_sample=True,
                top_p=0.92,
                temperature=0.85,
                min_length=60,
                no_repeat_ngram_size=4,
                repetition_penalty=1.45,
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # -------------------------------
        # Output
        # -------------------------------
        st.subheader("📌 Summary")
        st.write(summary)
