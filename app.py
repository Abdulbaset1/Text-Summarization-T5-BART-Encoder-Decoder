import streamlit as st
import os
import sys
import requests
import subprocess
import importlib

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

def install_missing_packages():
    """Install missing packages using pip"""
    packages_to_install = []
    
    # Check which packages are missing
    try:
        import torch
    except ImportError:
        packages_to_install.append("torch")
    
    try:
        import transformers
    except ImportError:
        packages_to_install.append("transformers")
    
    try:
        import tokenizers
    except ImportError:
        packages_to_install.append("tokenizers")
    
    try:
        import sentencepiece
    except ImportError:
        packages_to_install.append("sentencepiece")
    
    if packages_to_install:
        st.info(f"Installing missing packages: {', '.join(packages_to_install)}")
        try:
            for package in packages_to_install:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package,
                    "--user", "--no-warn-script-location"
                ])
            st.success("Packages installed successfully! Please refresh the page.")
            return True
        except Exception as e:
            st.error(f"Failed to install packages: {e}")
            return False
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    dependencies = {
        'torch': False,
        'transformers': False,
        'tokenizers': False,
        'sentencepiece': False
    }
    
    for dep in dependencies.keys():
        try:
            importlib.import_module(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def download_model():
    """Download the model file from GitHub releases"""
    model_url = "https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt"
    local_filename = "Finalmod.pt"
    
    if os.path.exists(local_filename):
        file_size = os.path.getsize(local_filename)
        if file_size > 100000:  # At least 100KB
            return local_filename
    
    try:
        st.info("📥 Downloading model file...")
        
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
                        if downloaded_size % (1024 * 1024) == 0:  # Update every MB
                            status_text.text(f"Downloaded: {downloaded_size/(1024*1024):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 100000:
            st.success("✅ Model downloaded successfully!")
            return local_filename
        else:
            st.error("❌ Download failed - file is too small")
            return None
            
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None

def simple_summarize(text, max_sentences=3):
    """Simple extractive summarization fallback"""
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    selected = [sentences[0]]  # First sentence
    if len(sentences) > 1:
        selected.append(sentences[len(sentences)//2])  # Middle sentence
    if len(sentences) > 2:
        selected.append(sentences[-1])  # Last sentence
    
    summary = '. '.join(selected[:max_sentences])
    return summary + '.' if not summary.endswith('.') else summary

@st.cache_resource
def load_ai_model():
    """Load AI model with caching"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
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
        import torch
        
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
        st.error(f"❌ AI summarization failed: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">📝 Text Summarization App</div>', unsafe_allow_html=True)
    st.markdown("### Using Fine-tuned T5 Model")
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Sidebar with dependency status
    with st.sidebar:
        st.markdown("## 🔧 Dependency Status")
        
        all_loaded = all(dependencies.values())
        
        if not all_loaded:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("Some AI dependencies are missing")
            st.markdown('</div>', unsafe_allow_html=True)
            
            for dep, loaded in dependencies.items():
                if loaded:
                    st.success(f"✅ {dep}")
                else:
                    st.error(f"❌ {dep}")
            
            st.markdown("---")
            if st.button("🔄 Install Missing Packages", use_container_width=True):
                if install_missing_packages():
                    st.rerun()
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.success("All AI dependencies loaded!")
            st.markdown('</div>', unsafe_allow_html=True)
            for dep, loaded in dependencies.items():
                st.success(f"✅ {dep}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app provides text summarization using:
        - 🤖 AI-Powered (T5 model)
        - 🔍 Basic extraction (fallback)
        
        **Features:**
        - Adjustable summary length
        - Text analysis
        - Download summaries
        """)
    
    # Load AI model if dependencies are available
    ai_model, ai_tokenizer, ai_device = None, None, None
    if all_loaded:
        ai_model, ai_tokenizer, ai_device = load_ai_model()
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        if dependencies['torch']:
            try:
                import torch
                st.success(f"**PyTorch:** {torch.__version__}")
            except:
                st.success("**PyTorch:** ✅ Available")
        else:
            st.warning("**PyTorch:** ❌ Missing")
    
    with col2:
        if ai_model:
            st.success("**AI Model:** ✅ Loaded")
        else:
            st.info("**AI Model:** 🔍 Basic Mode")
    
    with col3:
        st.info("**Status:** 🟢 Active")
    
    # Input section
    st.markdown("---")
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
        
        # Example texts
        st.markdown("**Try these examples:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Technology Article", use_container_width=True):
                st.session_state.demo_text = """
                Artificial intelligence is transforming various industries by automating complex tasks 
                and providing insights from large datasets. Machine learning algorithms can now 
                recognize patterns that were previously undetectable to humans. However, these 
                advancements also raise important ethical questions about privacy and job displacement. 
                Researchers are working on developing AI systems that are transparent and accountable.
                """
        with col2:
            if st.button("News Story", use_container_width=True):
                st.session_state.demo_text = """
                Scientists have discovered a new method for renewable energy storage that could 
                revolutionize how we power our cities. The breakthrough involves using advanced 
                materials to store solar energy more efficiently. This development comes at a 
                critical time as the world seeks solutions to climate change and energy security.
                """
        
        if hasattr(st.session_state, 'demo_text'):
            input_text = st.session_state.demo_text
            
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
    
    # Generate summary
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
                    method_used = "🤖 AI-Powered"
                    if not summary:
                        summary = simple_summarize(input_text, max_sentences=max_length//50)
                        method_used = "🔍 Basic Extraction (AI failed)"
                else:
                    summary = simple_summarize(input_text, max_sentences=max_length//50)
                    method_used = "🔍 Basic Extraction"
            
            if summary:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="original-box">', unsafe_allow_html=True)
                    st.markdown("### 📄 Original Text")
                    
                    display_text = input_text
                    if len(input_text) > 2000:
                        display_text = input_text[:2000] + "..."
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
                    
                    if "AI" in method_used:
                        st.success(method_used)
                    else:
                        st.info(method_used)
                    
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
                
                # Download button
                st.download_button(
                    label="💾 Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
