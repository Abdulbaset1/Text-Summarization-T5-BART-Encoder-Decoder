import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import os
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .original-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def download_model():
    """Download the model file from GitHub releases if not exists"""
    model_url = "https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt"
    local_filename = "Finalmod.pt"
    
    # Check if model already exists
    if os.path.exists(local_filename):
        st.success("✅ Model file found locally")
        return local_filename
    
    # Download with progress bar
    try:
        st.info("📥 Downloading model file (this may take a few minutes)...")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = min(downloaded_size / total_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded_size/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Model downloaded successfully!")
        return local_filename
        
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        # Download model first
        model_path = download_model()
        if not model_path:
            return None, None
        
        # Model configuration
        MODEL_NAME = "t5-small"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        
        # Load the trained model with progress
        st.info("🔄 Loading model into memory...")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device: {device}")
        
        # Initialize model architecture
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Load the trained weights
        model_weights = torch.load(model_path, map_location=device)
        model.load_state_dict(model_weights)
        model.to(device)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def generate_summary(model, tokenizer, device, text, max_length=150):
    """Generate summary for given text"""
    try:
        # Prepare input
        inputs = tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate summary
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=2.0
            )
        
        # Decode output
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">📝 Text Summarization App</div>', unsafe_allow_html=True)
    st.markdown("### Using Fine-tuned T5 Model")
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check the console for errors.")
        return
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["Type text", "Upload text file"],
        horizontal=True
    )
    
    input_text = ""
    
    if input_method == "Type text":
        input_text = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste your article, document, or any long text here...",
            help="Enter at least 100 characters for better results"
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'pdf', 'docx'])
        if uploaded_file is not None:
            # For simplicity, only handling txt files in this example
            if uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")
                st.text_area("Uploaded text preview:", input_text[:1000] + "..." if len(input_text) > 1000 else input_text, height=150)
            else:
                st.warning("Please upload a .txt file. Other formats coming soon!")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider(
            "Maximum summary length:",
            min_value=50,
            max_value=300,
            value=150,
            help="Longer summaries will be more detailed"
        )
    
    with col2:
        min_input_length = st.number_input(
            "Minimum input length:",
            min_value=50,
            max_value=1000,
            value=100,
            help="Skip summarization for very short texts"
        )
    
    with col3:
        show_analysis = st.checkbox("Show text analysis", value=True)
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to summarize.")
        elif len(input_text.strip()) < min_input_length:
            st.warning(f"⚠️ Please enter at least {min_input_length} characters for meaningful summarization.")
        else:
            with st.spinner("🔍 Analyzing text and generating summary..."):
                summary = generate_summary(model, tokenizer, device, input_text, max_length)
            
            if summary:
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="original-box">', unsafe_allow_html=True)
                    st.markdown("### 📄 Original Text")
                    st.write(input_text)
                    
                    if show_analysis:
                        st.markdown("---")
                        word_count = len(input_text.split())
                        char_count = len(input_text)
                        sentence_count = input_text.count('.') + input_text.count('!') + input_text.count('?')
                        
                        st.write("**📊 Text Analysis:**")
                        st.write(f"- **Words:** {word_count}")
                        st.write(f"- **Characters:** {char_count}")
                        st.write(f"- **Sentences:** {sentence_count}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown("### 📋 Generated Summary")
                    st.success(summary)
                    
                    if show_analysis:
                        st.markdown("---")
                        summary_word_count = len(summary.split())
                        summary_char_count = len(summary)
                        compression_ratio = (1 - summary_word_count/word_count) * 100 if word_count > 0 else 0
                        
                        st.write("**📈 Summary Analysis:**")
                        st.write(f"- **Words:** {summary_word_count}")
                        st.write(f"- **Characters:** {summary_char_count}")
                        st.write(f"- **Compression Ratio:** {compression_ratio:.1f}%")
                        
                        # Quality indicator
                        if compression_ratio > 80:
                            st.write("- **Quality:** ⭐ High compression")
                        elif compression_ratio > 50:
                            st.write("- **Quality:** ⭐⭐ Good balance")
                        else:
                            st.write("- **Quality:** ⭐⭐⭐ Detailed summary")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Summary",
                        data=summary,
                        file_name="generated_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="💾 Download Original + Summary",
                        data=f"ORIGINAL TEXT:\n{input_text}\n\nGENERATED SUMMARY:\n{summary}",
                        file_name="original_and_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

    # Sidebar with information
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This app uses a fine-tuned **T5 model** for text summarization.
        
        **✨ Features:**
        - Extract key information from long texts
        - Adjustable summary length
        - Support for text input and file upload
        - Text analysis and compression metrics
        - Download generated summaries
        
        **🚀 How to use:**
        1. Enter or upload your text
        2. Adjust summary length if needed
        3. Click 'Generate Summary'
        4. View and download the result
        """)
        
        st.markdown("---")
        st.markdown("**🔧 Model Info:**")
        st.markdown("- **Base Model:** T5-small")
        st.markdown("- **Task:** Text Summarization")
        st.markdown("- **Framework:** PyTorch")
        
        st.markdown("---")
        st.markdown("**📊 System Info:**")
        st.markdown(f"- **Device:** {'GPU 🔥' if torch.cuda.is_available() else 'CPU ⚡'}")
        st.markdown(f"- **PyTorch:** {torch.__version__}")
        
        # Clear cache button
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Reload the page to refresh.")

if __name__ == "__main__":
    main()
