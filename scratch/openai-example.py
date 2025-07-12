import os
import json
import requests
from typing import Dict, Any

def call_openai_compatible_endpoint(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    api_key: str = None,
    base_url: str = "http://localhost:8000/v1",  # Default to local DSP AI Gateway
    max_tokens: int = 100,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Make a request to an OpenAI-compatible API endpoint via DSP AI Gateway.
    
    Args:
        prompt: The text prompt to send to the model
        model: Model identifier (default: gpt-3.5-turbo)
        api_key: API key for authentication (if None, will look for DSP_API_KEY env var)
        base_url: Base URL for the API (default: local DSP AI Gateway)
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter for response randomness
        
    Returns:
        The API response as a dictionary
    """
    # Use environment variable if api_key is not provided
    if api_key is None:
        api_key = os.environ.get("DSP_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set as DSP_API_KEY environment variable")
    
    # Prepare headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Make the API call
    endpoint = f"{base_url}/chat/completions"
    response = requests.post(endpoint, headers=headers, json=payload)
    
    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()
    
    return response.json()

def main():
    # Example usage
    try:
        # Set your DSP AI Gateway API key here or as an environment variable
        # os.environ["DSP_API_KEY"] = "your-dsp-api-key-here"
        
        # DSP AI Gateway endpoint (modify as needed)
        base_url = "https://api.groq.com/openai/v1"  # Your DSP AI Gateway endpoint
        
        prompt = "Explain what an OpenAI-compatible API is in one sentence."
        
        print(f"Sending prompt to DSP AI Gateway: '{prompt}'")
        response = call_openai_compatible_endpoint(
            prompt=prompt,
            base_url=base_url,
            api_key="API_KEY",
            # You can specify different models available in your DSP AI Gateway
            model="llama-3.1-8b-instant"
        )
        
        # Extract and print the generated text
        generated_text = response["choices"][0]["message"]["content"]
        print("\nResponse:")
        print(generated_text)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()