import os
import sys

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: {e}")
    DEPS_AVAILABLE = False

class TextSummarizer:
    def __init__(self, model_path="https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        if DEPS_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
    
    def load_model(self):
        """Load the model and tokenizer"""
        if not DEPS_AVAILABLE:
            print("Required dependencies not available")
            return False
            
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} not found")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
            
            # Load model architecture
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            
            # Load trained weights
            model_weights = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(model_weights)
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.loaded = False
            return False
    
    def summarize(self, text, max_length=150):
        """Generate summary for given text"""
        if not self.loaded:
            if not self.load_model():
                return self._fallback_summarize(text)
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                "summarize: " + text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0
                )
            
            # Decode output
            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Error in AI summarization: {e}")
            return self._fallback_summarize(text)
    
    def _fallback_summarize(self, text, max_length=150):
        """Simple rule-based fallback summarization"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 2:
            return text
        
        # Simple extraction-based summary
        important_sentences = []
        if sentences:
            important_sentences.append(sentences[0])  # First sentence
        if len(sentences) > 1:
            important_sentences.append(sentences[len(sentences)//2])  # Middle sentence
        if len(sentences) > 2:
            important_sentences.append(sentences[-1])  # Last sentence
        
        summary = '. '.join(important_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        return summary

# For testing
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    test_text = """
    Artificial intelligence is transforming many aspects of our lives. From healthcare to transportation, 
    AI systems are being deployed to solve complex problems. Machine learning algorithms can analyze 
    vast amounts of data and identify patterns that humans might miss. However, there are also concerns 
    about job displacement and ethical considerations. Researchers are working on developing AI that 
    is transparent, fair, and beneficial for all of humanity.
    """
    
    result = summarizer.summarize(test_text)
    print("Summary:", result)
