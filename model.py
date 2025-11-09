import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self, model_path="Finalmod.pt"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        # Try to set device, but don't fail if torch not available
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            self.device = None
    
    def load_model(self):
        """Load the model if dependencies are available"""
        if not self.torch_available:
            logger.error("PyTorch not available")
            return False
            
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file {self.model_path} not found")
                return False
            
            # Load components
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            
            # Load weights
            model_weights = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(model_weights)
            self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def summarize(self, text, max_length=150):
        """Generate summary - falls back to simple method if AI not available"""
        if self.loaded:
            return self._ai_summarize(text, max_length)
        else:
            return self._simple_summarize(text)
    
    def _ai_summarize(self, text, max_length):
        """AI-powered summarization"""
        try:
            inputs = self.tokenizer(
                "summarize: " + text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with self.model._no_sync() if hasattr(self.model, '_no_sync') else nullcontext():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0
                )
            
            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"AI summarization failed: {e}")
            return self._simple_summarize(text)
    
    def _simple_summarize(self, text, max_sentences=3):
        """Simple extractive summarization"""
        import re
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        selected = [sentences[0]]
        if len(sentences) > 1:
            selected.append(sentences[len(sentences)//2])
        if len(sentences) > 2:
            selected.append(sentences[-1])
        
        summary = '. '.join(selected[:max_sentences])
        return summary + '.' if not summary.endswith('.') else summary

# Context manager for compatibility
class nullcontext:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass
