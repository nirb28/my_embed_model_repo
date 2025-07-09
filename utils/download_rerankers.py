#!/usr/bin/env python
"""
Script to download and save reranker models locally.
These reranker models will be served by the model server for use in the RAG pipeline.
"""

import os
import logging
import argparse
from pathlib import Path
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported reranker models
SUPPORTED_MODELS = {
    "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "ms-marco-tinybert": "cross-encoder/ms-marco-TinyBERT-L-2-v2"
}

def download_model(model_name, output_path):
    """
    Downloads a reranker model and saves it to the specified path.
    
    Args:
        model_name: Name of the model to download (from SUPPORTED_MODELS)
        output_path: Path to save the model to
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: {', '.join(SUPPORTED_MODELS.keys())}")
    
    hf_model_name = SUPPORTED_MODELS[model_name]
    target_dir = output_path / model_name
    
    if target_dir.exists():
        logger.warning(f"Model directory {target_dir} already exists. Skipping download.")
        return
    
    logger.info(f"Downloading model: {hf_model_name} to {target_dir}")
    
    # Load the model, which will download it from HuggingFace
    try:
        model = CrossEncoder(hf_model_name)
        
        # Create the target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Save the model to the target directory
        model.save(str(target_dir))
        logger.info(f"Successfully downloaded and saved model to {target_dir}")
    except Exception as e:
        logger.error(f"Error downloading model {hf_model_name}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download and save reranker models locally")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS.keys(), default=list(SUPPORTED_MODELS.keys()),
                      help=f"Models to download. Choose from: {', '.join(SUPPORTED_MODELS.keys())}")
    parser.add_argument("--output-path", type=str, default=str(Path(__file__).parent.parent / "models"),
                      help="Path to save the models")
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Download each requested model
    for model_name in args.models:
        try:
            download_model(model_name, output_path)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
    
    logger.info("Done downloading models")

if __name__ == "__main__":
    main()
