import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import logging
import requests
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self, model_path="Finalmod.pt"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.loaded = False
        logger.info(f"Initialized TextSummarizer with device: {self.device}")
        
        # Download model if not exists
        self._ensure_model_exists()
    
    def _ensure_model_exists(self):
        """Download the model file if it doesn't exist"""
        if not os.path.exists(self.model_path):
            logger.info("Model file not found. Downloading...")
            self._download_model()
        else:
            logger.info("Model file found locally")
    
    def _download_model(self):
        """Download the model from GitHub releases"""
        try:
            # CORRECT direct download URL for your model
            model_url = "https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt"
            
            logger.info(f"Downloading model from: {model_url}")
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Log progress for large files
                        if total_size > 0 and downloaded_size % (1024 * 1024) == 0:
                            progress = (downloaded_size / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info("✅ Model downloaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error downloading model: {str(e)}")
            # Try alternative URL pattern
            self._try_alternative_download()
    
    def _try_alternative_download(self):
        """Try alternative download URL patterns"""
        alternative_urls = [
            # Direct asset URL
            "https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt",
            # Raw GitHub URL (if file is in repo)
            "https://raw.githubusercontent.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/main/Finalmod.pt",
        ]
        
        for url in alternative_urls:
            try:
                logger.info(f"Trying alternative URL: {url}")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(self.model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logger.info(f"✅ Model downloaded from alternative URL!")
                    return
            except Exception as e:
                logger.error(f"Failed with URL {url}: {str(e)}")
        
        logger.error("❌ All download attempts failed!")
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} not found after download attempts")
            
            # Check file size to ensure download was successful
            file_size = os.path.getsize(self.model_path)
            logger.info(f"Model file size: {file_size} bytes")
            
            if file_size < 1000:  # If file is too small, download probably failed
                logger.warning("Model file seems too small, re-downloading...")
                os.remove(self.model_path)
                self._download_model()
            
            logger.info("Loading tokenizer...")
            # Load tokenizer - use the same as in your training
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
            
            logger.info("Loading model architecture...")
            # Load model architecture
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            
            logger.info("Loading trained weights...")
            # Load trained weights with proper error handling
            try:
                # Try loading with the standard approach first
                model_weights = torch.load(self.model_path, map_location=self.device)
                
                # Check if it's a state dict or full model
                if isinstance(model_weights, dict):
                    # It's a state dictionary
                    self.model.load_state_dict(model_weights)
                else:
                    # It might be a full model
                    self.model = model_weights
                    
            except Exception as e:
                logger.error(f"Error loading weights: {e}")
                # Try alternative loading methods
                try:
                    # Try loading with weights_only=False for older PyTorch versions
                    model_weights = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    if isinstance(model_weights, dict):
                        self.model.load_state_dict(model_weights)
                    else:
                        self.model = model_weights
                except Exception as e2:
                    logger.error(f"Alternative loading also failed: {e2}")
                    raise e2
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info("✅ Model loaded successfully!")
            
            # Test the model with a simple input
            test_result = self._test_model()
            if test_result:
                logger.info("✅ Model test passed!")
            else:
                logger.warning("⚠️ Model test failed!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self.loaded = False
            return False
    
    def _test_model(self):
        """Test the model with a simple input to ensure it's working"""
        try:
            test_text = "This is a test sentence for model verification."
            test_summary = self.summarize(test_text, max_length=50)
            return test_summary is not None and len(test_summary) > 0
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False
    
    def summarize(self, text, max_length=150, min_length=30, num_beams=4, temperature=0.8, early_stopping=True):
        """Generate summary for given text"""
        if not self.loaded:
            success = self.load_model()
            if not success:
                return "Error: Model not loaded properly. Please check if the model file is available."
        
        try:
            # Clean and prepare input text
            text = text.strip()
            if not text:
                return "Error: Input text is empty"
            
            # Prepare input with the same format used during training
            input_text = "summarize: " + text
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="longest",
                add_special_tokens=True
            ).to(self.device)
            
            # Generate summary with the same parameters as training
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0,
                    temperature=temperature,
                    do_sample=temperature != 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Clean up the summary
            summary = summary.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error during summarization: {str(e)}"
    
    def batch_summarize(self, texts, max_length=150):
        """Generate summaries for multiple texts"""
        if not self.loaded:
            self.load_model()
        
        summaries = []
        for i, text in enumerate(texts):
            try:
                summary = self.summarize(text, max_length)
                summaries.append(summary)
                logger.info(f"Processed text {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Error processing text {i+1}: {e}")
                summaries.append(f"Error: {str(e)}")
        
        return summaries

    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.loaded:
            return "Model not loaded"
        
        info = {
            "model_type": type(self.model).__name__,
            "device": str(self.device),
            "tokenizer": type(self.tokenizer).__name__,
            "vocab_size": len(self.tokenizer) if self.tokenizer else "N/A",
        }
        
        return info

    def __del__(self):
        """Clean up GPU memory if using CUDA"""
        if hasattr(self, 'model') and self.model is not None:
            if torch.cuda.is_available():
                self.model.cpu()
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")

# Utility function for quick testing
def test_summarization():
    """Test function to verify the summarizer works"""
    summarizer = TextSummarizer()
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    natural intelligence displayed by animals including humans. Leading AI textbooks 
    define the field as the study of intelligent agents: any system that perceives its 
    environment and takes actions that maximize its chance of achieving its goals. 
    Some popular accounts use the term artificial intelligence to describe machines that 
    mimic cognitive functions that humans associate with the human mind, such as learning 
    and problem solving. The field was founded on the assumption that human intelligence 
    can be so precisely described that it can be simulated by a machine.
    """
    
    print("Testing summarization...")
    print("Original text length:", len(sample_text))
    
    summary = summarizer.summarize(sample_text)
    print("Summary:", summary)
    print("Summary length:", len(summary))
    
    # Print model info
    info = summarizer.get_model_info()
    print("Model info:", info)

if __name__ == "__main__":
    test_summarization()
