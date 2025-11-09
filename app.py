import streamlit as st
import os
import sys
import requests
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Add current directory to path to import model
sys.path.append(os.path.dirname(__file__))

# Try to import torch and model
try:
    import torch
    from model import TextSummarizer
    DEPS_AVAILABLE = True
except ImportError as e:
    st.error(f"Required packages are loading: {e}")
    DEPS_AVAILABLE = False

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
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def download_model_from_github():
    """Download the model file from GitHub releases"""
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
def load_summarizer():
    """Load the text summarizer with cached resource"""
    if not DEPS_AVAILABLE:
        st.error("Required packages not available. Please check the dependencies.")
        return None
    
    try:
        # Download model first
        model_path = download_model_from_github()
        if not model_path:
            return None
        
        # Initialize and load the summarizer
        st.info("🔄 Loading model into memory...")
        summarizer = TextSummarizer(model_path=model_path)
        
        # Load the model
        success = summarizer.load_model()
        if success:
            st.success("✅ Model loaded successfully!")
            return summarizer
        else:
            st.error("❌ Failed to load model")
            return None
            
    except Exception as e:
        st.error(f"❌ Error initializing summarizer: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">📝 Text Summarization App</div>', unsafe_allow_html=True)
    st.markdown("### Using Your Fine-tuned T5 Model")
    
    # Display system info
    if DEPS_AVAILABLE:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**PyTorch:** {torch.__version__}")
        with col2:
            device = "GPU 🔥" if torch.cuda.is_available() else "CPU ⚡"
            st.info(f"**Device:** {device}")
        with col3:
            st.info("**Model:** T5-small Fine-tuned")
    
    # Load the summarizer
    with st.spinner("Loading summarization model..."):
        summarizer = load_summarizer()
    
    if summarizer is None:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("""
        ⚠️ **Model is still loading or unavailable**
        
        Please wait while the model loads. If this persists, try refreshing the page.
        The model file is being downloaded from GitHub releases.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
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
        min_length = st.slider(
            "Minimum summary length:",
            min_value=10,
            max_value=100,
            value=30,
            help="Shorter summaries will be more concise"
        )
    
    with col3:
        show_analysis = st.checkbox("Show text analysis", value=True)
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_beams = st.slider(
                "Number of beams:",
                min_value=1,
                max_value=8,
                value=4,
                help="Higher values can improve quality but slow down generation"
            )
        with col2:
            temperature = st.slider(
                "Temperature:",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                help="Lower values make output more deterministic"
            )
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to summarize.")
        elif len(input_text.strip()) < 50:
            st.warning("⚠️ Please enter at least 50 characters for meaningful summarization.")
        else:
            with st.spinner("🔍 Analyzing text and generating summary..."):
                try:
                    summary = summarizer.summarize(
                        input_text, 
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        temperature=temperature
                    )
                except Exception as e:
                    st.error(f"❌ Error during summarization: {str(e)}")
                    summary = None
            
            if summary:
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="original-box">', unsafe_allow_html=True)
                    st.markdown("### 📄 Original Text")
                    
                    # Show truncated text for long inputs
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
        This app uses your **fine-tuned T5 model** for text summarization.
        
        **✨ Features:**
        - Uses your custom trained model
        - Adjustable summary length
        - Advanced generation parameters
        - Text analysis and metrics
        - Download functionality
        
        **🚀 How to use:**
        1. Enter or upload your text
        2. Adjust parameters if needed
        3. Click 'Generate Summary'
        4. View and download results
        """)
        
        st.markdown("---")
        st.markdown("**🔧 Model Info:**")
        st.markdown("- **Base Model:** T5-small")
        st.markdown("- **Training:** Custom fine-tuned")
        st.markdown("- **Framework:** PyTorch")
        
        st.markdown("---")
        if DEPS_AVAILABLE:
            st.markdown("**📊 System Info:**")
            st.markdown(f"- **Device:** {'GPU 🔥' if torch.cuda.is_available() else 'CPU ⚡'}")
            st.markdown(f"- **PyTorch:** {torch.__version__}")
        
        # Clear cache button
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared! Reload the page.")

if __name__ == "__main__":
    main()
