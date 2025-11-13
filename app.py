import streamlit as st
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the LoRA model and tokenizer"""
    try:
        # Your model details
        base_model = "t5-small"
        peft_model_id = "basit1878/t5-small-lora-summarizer"  # Your model
        
        # Load base model
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, peft_model_id)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def summarize_text(text, model, tokenizer, max_length=150):
    """Generate summary for the input text"""
    try:
        # Prepare input
        inputs = tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=2.5,
                length_penalty=1.0
            )
        
        # Decode output
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

# Main app
def main():
    st.title("📝 AI Text Summarizer")
    st.markdown("""
    This app uses a fine-tuned T5 model with LoRA to generate concise summaries of your text.
    **Model:** [basit1878/t5-small-lora-summarizer](https://huggingface.co/basit1878/t5-small-lora-summarizer)
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    max_length = st.sidebar.slider("Summary Length", 50, 250, 150, 10)
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Paste your text in the box below
    2. Click 'Generate Summary'
    3. Get your concise summary!
    """)
    
    # Load model
    with st.spinner("Loading model... This may take a minute."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check the connection.")
        return
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to summarize:",
            height=300,
            placeholder="Paste your article, document, or any long text here..."
        )
    
    with col2:
        st.subheader("Example Text")
        st.markdown("""
        **Sample input:**
        ```
        The quick brown fox jumps over the lazy dog. 
        This sentence contains all letters in the alphabet. 
        It's often used for typing practice and font demonstrations.
        ```
        """)
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary"):
        if text_input.strip():
            with st.spinner("Generating summary..."):
                summary = summarize_text(text_input, model, tokenizer, max_length)
            
            st.success("Summary Generated!")
            st.subheader("📄 Summary:")
            st.write(summary)
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{len(text_input.split())} words")
            with col2:
                st.metric("Summary Length", f"{len(summary.split())} words")
            with col3:
                reduction = ((len(text_input.split()) - len(summary.split())) / len(text_input.split())) * 100
                st.metric("Reduction", f"{reduction:.1f}%")
        else:
            st.warning("Please enter some text to summarize.")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Model**: T5-small with LoRA fine-tuning | "
        "**Built with**: Streamlit & Hugging Face"
    )

if __name__ == "__main__":
    main()
