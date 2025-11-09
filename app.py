import streamlit as st
import os
import sys
import requests
import time

# Set page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

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
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
        st.session_state.torch_available = True
    except ImportError:
        missing_deps.append("torch")
        st.session_state.torch_available = False
        
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        st.session_state.transformers_available = True
    except ImportError:
        missing_deps.append("transformers")
        st.session_state.transformers_available = False
        
    return missing_deps

def download_model():
    """Download the model file from GitHub releases"""
    model_url = "https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt"
    local_filename = "Finalmod.pt"
    
    # Check if model already exists
    if os.path.exists(local_filename):
        file_size = os.path.getsize(local_filename)
        if file_size > 1000:  # Basic check if file is not empty/corrupted
            st.success(f"✅ Model file found ({file_size} bytes)")
            return local_filename
        else:
            st.warning("⚠️ Model file seems corrupted, re-downloading...")
            os.remove(local_filename)
    
    # Download the model
    try:
        st.info("📥 Downloading model file... This may take a few minutes.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = min(downloaded_size / total_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded_size/(1024*1024):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        # Verify download
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1000:
            st.success("✅ Model downloaded successfully!")
            return local_filename
        else:
            st.error("❌ Download failed - file is too small or corrupted")
            return None
            
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None

def simple_summarize(text, max_sentences=3):
    """Simple extractive summarization as fallback"""
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    # Take important sentences (first, middle, last)
    selected_indices = [0]  # First sentence
    
    if len(sentences) > 1:
        selected_indices.append(len(sentences) // 2)  # Middle sentence
    
    if len(sentences) > 2:
        selected_indices.append(-1)  # Last sentence
    
    selected_indices = selected_indices[:max_sentences]
    summary_sentences = [sentences[i] for i in selected_indices if 0 <= i < len(sentences)]
    
    summary = '. '.join(summary_sentences)
    if summary and not summary.endswith('.'):
        summary += '.'
    
    return summary

@st.cache_resource
def load_ai_model():
    """Load the AI model with caching"""
    if not st.session_state.torch_available or not st.session_state.transformers_available:
        return None, None, None
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Download model first
        model_path = download_model()
        if not model_path:
            return None, None, None
        
        st.info("🔄 Loading AI model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
        
        # Load model architecture
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        
        # Load trained weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights = torch.load(model_path, map_location=device)
        model.load_state_dict(model_weights)
        model.to(device)
        model.eval()
        
        st.success("✅ AI model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading AI model: {str(e)}")
        return None, None, None

def generate_ai_summary(model, tokenizer, device, text, max_length=150):
    """Generate summary using AI model"""
    try:
        inputs = tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=2.0
            )
        
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"❌ AI summarization failed: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">📝 Text Summarization App</div>', unsafe_allow_html=True)
    st.markdown("### Using Fine-tuned T5 Model")
    
    # Initialize session state
    if 'deps_checked' not in st.session_state:
        st.session_state.deps_checked = False
        st.session_state.torch_available = False
        st.session_state.transformers_available = False
    
    # Check dependencies
    if not st.session_state.deps_checked:
        with st.spinner("Checking dependencies..."):
            missing_deps = check_dependencies()
            st.session_state.deps_checked = True
            
            if missing_deps:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
                st.info("Using basic summarization method. AI features will be available when dependencies are loaded.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.success("✅ All dependencies loaded successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Load AI model if dependencies are available
    ai_model, ai_tokenizer, ai_device = None, None, None
    if st.session_state.torch_available and st.session_state.transformers_available:
        ai_model, ai_tokenizer, ai_device = load_ai_model()
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.torch_available:
            import torch
            st.info(f"**PyTorch:** {torch.__version__}")
        else:
            st.warning("**PyTorch:** Loading...")
    
    with col2:
        if ai_model:
            st.success("**AI Model:** ✅ Ready")
        else:
            st.warning("**AI Model:** ⚠️ Basic mode")
    
    with col3:
        st.info("**Summarization:** Available")
    
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
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file is not None:
            try:
                input_text = uploaded_file.read().decode("utf-8")
                st.text_area("Uploaded text preview:", 
                           input_text[:1000] + "..." if len(input_text) > 1000 else input_text, 
                           height=150)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider(
            "Summary length:",
            min_value=50,
            max_value=300,
            value=150,
            help="Adjust the length of the summary"
        )
    
    with col2:
        show_analysis = st.checkbox("Show text analysis", value=True)
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to summarize.")
        elif len(input_text.strip()) < 50:
            st.warning("⚠️ Please enter at least 50 characters.")
        else:
            with st.spinner("🔍 Analyzing text and generating summary..."):
                # Try AI model first, fallback to simple method
                if ai_model and ai_tokenizer and ai_device:
                    summary = generate_ai_summary(ai_model, ai_tokenizer, ai_device, input_text, max_length)
                    method_used = "AI-Powered"
                else:
                    summary = simple_summarize(input_text, max_sentences=max_length//50)
                    method_used = "Basic Extraction"
                
                if not summary:
                    summary = simple_summarize(input_text, max_sentences=max_length//50)
                    method_used = "Basic Extraction (AI failed)"
            
            if summary:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="original-box">', unsafe_allow_html=True)
                    st.markdown("### 📄 Original Text")
                    
                    display_text = input_text
                    if len(input_text) > 2000:
                        display_text = input_text[:2000] + "...\n\n[Text truncated for display]"
                    st.write(display_text)
                    
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
                    
                    if method_used == "AI-Powered":
                        st.success("🤖 " + method_used)
                    else:
                        st.info("🔍 " + method_used)
                    
                    st.write(summary)
                    
                    if show_analysis:
                        st.markdown("---")
                        summary_word_count = len(summary.split())
                        summary_char_count = len(summary)
                        compression_ratio = (1 - summary_word_count/word_count) * 100 if word_count > 0 else 0
                        
                        st.write("**📈 Summary Analysis:**")
                        st.write(f"- **Words:** {summary_word_count}")
                        st.write(f"- **Characters:** {summary_char_count}")
                        st.write(f"- **Compression:** {compression_ratio:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download buttons
                st.download_button(
                    label="💾 Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This app provides text summarization using:
        
        - **AI-Powered**: Fine-tuned T5 model
        - **Basic**: Extract key sentences
        
        **Current Mode:** {}
        """.format("🤖 AI-Powered" if ai_model else "🔍 Basic"))
        
        st.markdown("---")
        st.markdown("**🔄 Refresh Status**")
        if st.button("Check Dependencies Again"):
            st.session_state.deps_checked = False
            st.rerun()

if __name__ == "__main__":
    main()
