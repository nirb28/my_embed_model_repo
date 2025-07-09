#!/usr/bin/env python
"""
Test script for verifying the reranker models through the model server.
"""

import requests
import json
import logging
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reranker(server_url, model_name="ms-marco-minilm"):
    """Test the reranker endpoint with a query and sample documents"""
    endpoint = f"{server_url}/rerank"
    
    # Sample query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time.",
        "Python is a popular programming language used for various purposes like web development, data analysis, and artificial intelligence.",
        "The weather forecast predicts rain for tomorrow afternoon.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Natural language processing is concerned with the interactions between computers and human language."
    ]
    
    # Prepare request payload
    payload = {
        "query": query,
        "documents": documents,
        "model_name": model_name
    }
    
    logger.info(f"Testing reranker endpoint at {endpoint} with model {model_name}")
    logger.info(f"Query: {query}")
    logger.info(f"Documents: {len(documents)}")
    
    # Make request to the rerank endpoint
    try:
        response = requests.post(
            url=endpoint,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return False
            
        result = response.json()
        scores = result["scores"]
        
        # Print results
        logger.info("Reranking Results:")
        for i, (doc, score) in enumerate(zip(documents, scores)):
            logger.info(f"Document {i+1}: Score={score:.4f}")
            logger.info(f"  Text: {doc[:100]}...")
        
        # Sort documents by score
        sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        logger.info("\nRanked Documents:")
        for i, (doc, score) in enumerate(sorted_docs):
            logger.info(f"{i+1}. Score={score:.4f}: {doc[:100]}...")
        
        logger.info("Reranking test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error testing reranker: {str(e)}")
        return False

def main():
    parser = ArgumentParser(description="Test the reranker model server endpoint")
    parser.add_argument("--server-url", type=str, default="http://localhost:9001", 
                        help="URL of the model server")
    parser.add_argument("--model-name", type=str, default="ms-marco-minilm", 
                        help="Name of the reranker model to test")
    
    args = parser.parse_args()
    test_reranker(args.server_url, args.model_name)

if __name__ == "__main__":
    main()
