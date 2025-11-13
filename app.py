import streamlit as st
import requests
import json

# Set page config
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Hugging Face API details
API_URL = " https://router.huggingface.co/hf-inference/models/basit1878/t5-small-lora-summarizer"
headers = {"Authorization": "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"}

def query_huggingface(payload):
    """Send request to Hugging Face API"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("📝 AI Text Summarizer")
    st.markdown("""
    This app uses a fine-tuned T5 model with LoRA to generate concise summaries of your text.
    **Model:** [basit1878/t5-small-lora-summarizer](https://huggingface.co/basit1878/t5-small-lora-summarizer)
    """)
    
    # Input section
    text_input = st.text_area(
        "Enter text to summarize:",
        height=300,
        placeholder="Paste your article, document, or any long text here..."
    )
    
    # Instructions for getting API token
    with st.expander("🔑 How to get API Token"):
        st.markdown("""
        1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
        2. Create a new token with **Write** access
        3. Replace `YOUR_HF_TOKEN_HERE` in the code with your actual token
        4. Redeploy the app
        """)
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary"):
        if text_input.strip():
            with st.spinner("Generating summary..."):
                # Prepare payload
                payload = {
                    "inputs": "summarize: " + text_input,
                    "parameters": {
                        "max_length": 150,
                        "min_length": 30,
                        "do_sample": False
                    }
                }
                
                # Call Hugging Face API
                result = query_huggingface(payload)
                
                # Display result
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get('generated_text', 'No summary generated')
                    st.success("Summary Generated!")
                    st.subheader("📄 Summary:")
                    st.write(summary)
                else:
                    st.error(f"Error: {result}")
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
