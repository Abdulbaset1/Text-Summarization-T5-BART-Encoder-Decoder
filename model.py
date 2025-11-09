
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

class TextSummarizer:
    def __init__(self, model_path="https://github.com/Abdulbaset1/Text-Summarization-T5-BART-Encoder-Decoder/releases/tag/v1/Finalmod.pt"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self):
        """Load the model and tokenizer"""
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
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.loaded = False
    
    def summarize(self, text, max_length=150, min_length=30):
        """Generate summary for given text"""
        if not self.loaded:
            self.load_model()
            if not self.loaded:
                return "Error: Model not loaded properly"
        
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
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0,
                    temperature=0.8
                )
            
            # Decode output
            summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def batch_summarize(self, texts, max_length=150):
        """Generate summaries for multiple texts"""
        summaries = []
        for text in texts:
            summary = self.summarize(text, max_length)
            summaries.append(summary)
        return summaries

# Example usage
if __name__ == "__main__":
    # Test the model
    summarizer = TextSummarizer()
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
    natural intelligence displayed by animals including humans. Leading AI textbooks 
    define the field as the study of intelligent agents: any system that perceives its 
    environment and takes actions that maximize its chance of achieving its goals. 
    Some popular accounts use the term artificial intelligence to describe machines that 
    mimic cognitive functions that humans associate with the human mind, such as learning 
    and problem solving.
    """
    
    summary = summarizer.summarize(sample_text)
    print("Original:", sample_text)
    print("Summary:", summary)
