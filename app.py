import streamlit as st
import requests
import json
import os
from typing import Optional

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

class SummarizationService:
    def __init__(self):
        self.methods = ["huggingface_api", "extractive_fallback"]
        self.current_method = "extractive_fallback"
        
    def summarize_with_huggingface(self, text: str, max_length: int = 150) -> Optional[str]:
        """Use Hugging Face Inference API for summarization"""
        try:
            # Using a free model from Hugging Face
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": "Bearer hf_your_token_here"}  # You can get a free token
            
            payload = {
                "inputs": text,
                "parameters": {
                    "max_length": max_length,
                    "min_length": 30,
                    "do_sample": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0]['summary_text']
            return None
            
        except Exception as e:
            st.sidebar.warning(f"Hugging Face API unavailable: {e}")
            return None
    
    def extractive_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization as fallback"""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple algorithm: take first, middle, and last sentences
        selected_indices = [0]  # First sentence
        
        if len(sentences) > 1:
            selected_indices.append(len(sentences) // 2)  # Middle sentence
        
        if len(sentences) > 2:
            selected_indices.append(-1)  # Last sentence
        
        # Ensure we don't exceed max_sentences
        selected_indices = selected_indices[:max_sentences]
        
        summary_sentences = []
        for idx in selected_indices:
            if 0 <= idx < len(sentences):
                summary_sentences.append(sentences[idx])
        
        summary = '. '.join(summary_sentences)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Main summarization method with fallbacks"""
        # Try Hugging Face API first
        summary = self.summarize_with_huggingface(text, max_length)
        if summary:
            self.current_method = "huggingface_api"
            return summary
        
        # Fallback to extractive summarization
        self.current_method = "extractive_fallback"
        return self.extractive_summarize(text, max_sentences=max_length//50)

def initialize_summarizer():
    """Initialize the summarization service"""
    return SummarizationService()

def main():
    # Header
    st.markdown('<div class="main-header">📝 Text Summarization App</div>', unsafe_allow_html=True)
    st.markdown("### Smart Text Summarization")
    
    # Initialize service
    summarizer = initialize_summarizer()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This app provides intelligent text summarization using multiple methods.
        
        **✨ Features:**
        - AI-powered summarization
        - Adjustable summary length
        - Support for text input and file upload
        - Text analysis and compression metrics
        - Download generated summaries
        
        **🔧 Current Method:**
        - Extractive Summarization
        - (API methods available with configuration)
        """)
        
        st.markdown("---")
        st.markdown("**💡 Pro Tip:**")
        st.markdown("For longer documents, break them into sections for better results.")
        
        st.markdown("---")
        st.markdown("**🔧 Advanced Options**")
        use_advanced = st.checkbox("Show advanced options", value=False)
        
        if use_advanced:
            st.info("""
            To use AI-powered summarization:
            1. Get a free API token from Hugging Face
            2. Add it to the code in app.py
            3. The app will automatically use the AI model
            """)
    
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
        
        # Example texts for quick testing
        st.markdown("**Quick examples:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Tech Article", use_container_width=True):
                st.session_state.example_text = """
                Artificial intelligence is transforming the way we interact with technology. 
                Recent advances in machine learning have enabled computers to understand and 
                generate human-like text, opening up new possibilities for automation and 
                creativity. However, these developments also raise important ethical questions 
                about privacy, bias, and the future of work. Researchers are actively working 
                on making AI systems more transparent and accountable.
                """
        
        with col2:
            if st.button("News Story", use_container_width=True):
                st.session_state.example_text = """
                Scientists have made a breakthrough in renewable energy technology. 
                A new solar panel design has achieved record efficiency levels, 
                potentially making solar power more affordable and accessible. 
                The innovation could help accelerate the transition to clean energy 
                and combat climate change. Further testing is needed before mass production.
                """
        
        with col3:
            if st.button("Research Paper", use_container_width=True):
                st.session_state.example_text = """
                This study examines the impact of social media on mental health. 
                Through a comprehensive analysis of user behavior and psychological 
                assessments, we found correlations between excessive social media use 
                and increased anxiety levels. The research suggests the need for 
                digital wellness tools and more balanced online engagement practices.
                """
        
        # Apply example text if selected
        if hasattr(st.session_state, 'example_text'):
            input_text = st.session_state.example_text
            
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
            "Summary length:",
            min_value=50,
            max_value=300,
            value=150,
            help="Adjust the length of the generated summary"
        )
    
    with col2:
        summary_type = st.selectbox(
            "Summary style:",
            ["Concise", "Balanced", "Detailed"],
            help="Choose how detailed you want the summary to be"
        )
    
    with col3:
        show_analysis = st.checkbox("Show analysis", value=True)
    
    # Map summary type to parameters
    type_to_sentences = {
        "Concise": 2,
        "Balanced": 3,
        "Detailed": 4
    }
    
    # Generate button
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to summarize.")
        elif len(input_text.strip()) < 50:
            st.warning("⚠️ Please enter at least 50 characters for meaningful summarization.")
        else:
            with st.spinner("🔍 Analyzing text and generating summary..."):
                # Adjust parameters based on selection
                if summary_type != "Balanced":
                    max_sentences = type_to_sentences[summary_type]
                    custom_max_length = max_length
                else:
                    max_sentences = 3
                    custom_max_length = max_length
                
                summary = summarizer.summarize(input_text, custom_max_length)
            
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
                        
                        # Readability estimate
                        if word_count > 0 and sentence_count > 0:
                            avg_sentence_length = word_count / sentence_count
                            if avg_sentence_length < 15:
                                readability = "Easy"
                            elif avg_sentence_length < 25:
                                readability = "Moderate"
                            else:
                                readability = "Complex"
                            st.write(f"- **Readability:** {readability}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown("### 📋 Generated Summary")
                    
                    # Color code based on method
                    if summarizer.current_method == "huggingface_api":
                        st.success("🤖 AI-Powered Summary")
                    else:
                        st.info("🔍 Smart Extract Summary")
                    
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
                        
                        # Quality indicator
                        if compression_ratio > 80:
                            quality = "⭐ High compression"
                        elif compression_ratio > 50:
                            quality = "⭐⭐ Good balance"
                        else:
                            quality = "⭐⭐⭐ Detailed"
                        st.write(f"- **Style:** {quality}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="💾 Full Report",
                        data=f"ORIGINAL TEXT:\n{input_text}\n\nSUMMARY:\n{summary}",
                        file_name="full_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Tips for improvement
                with st.expander("💡 Tips for better summaries"):
                    st.markdown("""
                    - **For longer documents:** Break into sections of 500-1000 words
                    - **For better accuracy:** Ensure the text is well-structured with clear sentences
                    - **For technical content:** Consider summarizing section by section
                    - **Current method:** {}
                    """.format("AI-Powered (BART model)" if summarizer.current_method == "huggingface_api" else "Extractive (Key sentences)"))

if __name__ == "__main__":
    main()
