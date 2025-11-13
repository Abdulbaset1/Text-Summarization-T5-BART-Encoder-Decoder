import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

st.title("📄 T5 LoRA Summarizer")
st.write("Model: basit1878/t5-small-lora-summarizer")

# Load model and tokenizer
@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, "basit1878/t5-small-lora-summarizer")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return model, tokenizer

model, tokenizer = load_model()

# Input text
text = st.text_area("Enter text to summarize:", height=250)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=1.4,
                min_length=60,
                length_penalty=1.2
            )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Summary")
        st.write(summary)
