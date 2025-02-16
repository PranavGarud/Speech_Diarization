import os
import sys
import codecs
from transformers import Wav2Vec2Processor, AutoModelForCTC

# Set UTF-8 encoding for stdout
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Set up paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_id = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
save_path = os.path.join(BASE_DIR, "Models", "Hindi model", "vakyansh-wav2vec2-hindi-him-4200")

print(f"Downloading Hindi model to: {save_path}")

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

if __name__ == "__main__":
    try:
        # Using Wav2Vec2Processor instead of AutoProcessor
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        
        # Download model in both formats
        model = AutoModelForCTC.from_pretrained(model_id, use_safetensors=True)
        model.save_pretrained(save_path, safe_serialization=True)  # Save safetensors
        model.save_pretrained(save_path, safe_serialization=False)  # Save pytorch_model.bin
        
        # Save processor
        processor.save_pretrained(save_path)
        
        print("Hindi model downloaded successfully!")
        
        # Verify both model files are present
        required_files = [
            "config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
            "model.safetensors",
            "pytorch_model.bin"
        ]
        
        print("\nVerifying downloaded files:")
        for file in required_files:
            file_path = os.path.join(save_path, file)
            if os.path.exists(file_path):
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")

    except Exception as e:
        print(f"Error downloading model: {str(e)}") 
