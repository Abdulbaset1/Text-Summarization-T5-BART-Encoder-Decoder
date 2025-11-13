import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

st.set_page_config(page_title="T5 Summarizer", layout="wide")
st.title("📄 T5 Summarizer — LoRA Fine-Tuned Model")
st.write("Model: **basit1878/t5-small-lora-summarizer**")

@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, "basit1878/t5-small-lora-summarizer")

    # Merge LoRA into base model
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    return model, tokenizer

model, tokenizer = load_model()

txt = st.text_area("Enter text to summarize:", height=250)

if st.button("Summarize"):
    if not txt.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            "summarize: " + txt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Summary")
        st.write(summary)
