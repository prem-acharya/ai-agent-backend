import json
import requests

def test_chat_endpoint(message: str, websearch: bool = False):
    url = "http://localhost:8000/api/v1/chat"
    
    data = {
        "content": message,
        "websearch": websearch
    }

    headers = {
        "Content-type": "application/json",
        "Accept": "text/event-stream"
    }

    try:
        print("\n" + "="*50)
        print(f"🔍 Query: {message}")
        print(f"🌐 Web Search: {'Enabled' if websearch else 'Disabled'}")
        print("="*50 + "\n")
        
        with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
            r.raise_for_status()
            print("Response streaming:")
            print("-" * 50)
            
            # Process streaming response
            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    try:
                        # Try to decode as JSON if possible
                        content = chunk.decode('utf-8')
                        print(content, end='', flush=True)
                    except Exception:
                        # If not JSON, print raw content
                        print(chunk, end='', flush=True)
            
            print("\n" + "-" * 50)

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    
    message = "latest bollywood movies and series are released"
    
    # Test with websearch enabled
    test_chat_endpoint(message, websearch=False)