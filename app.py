import streamlit as st

st.title("Text Summarizer")
st.write("Loading...")

try:
    import torch
    st.write("✓ Torch installed")
except:
    st.write("✗ Torch missing")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    st.write("✓ Transformers installed")
except:
    st.write("✗ Transformers missing")

try:
    from peft import PeftModel
    st.write("✓ PEFT installed")
except:
    st.write("✗ PEFT missing")
