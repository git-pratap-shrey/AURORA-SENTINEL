"""
Test script for chat/question-answering functionality
"""
import requests
import base64
import json
from PIL import Image
from io import BytesIO

def test_chat():
    """Test the chat endpoint with sample questions"""
    
    # Create a test image (or load from file)
    print("Creating test image...")
    test_image = Image.new('RGB', (640, 480), color='blue')
    
    # Add some text to make it more interesting
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(test_image)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text((100, 200), "TEST IMAGE", fill='white', font=font)
    
    # Convert to base64
    buffer = BytesIO()
    test_image.save(buffer, format='JPEG')
    image_data = base64.b64encode(buffer.getvalue()).decode()
    image_data = f"data:image/jpeg;base64,{image_data}"
    
    # Test questions
    questions = [
        "What do you see in this image?",
        "Describe the scene",
        "Is there any text in the image?",
        "What color is the background?",
        "Are there any people in this image?"
    ]
    
    print("\n" + "="*60)
    print("Testing Chat/Question-Answering Endpoint")
    print("="*60 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)
        
        try:
            response = requests.post('http://localhost:3001/chat', json={
                'imageData': image_data,
                'question': question
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Provider: {result['provider']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Request failed: {e}")
    
    print("\n" + "="*60)
    print("Chat test complete!")
    print("="*60)

if __name__ == "__main__":
    print("Chat/Question-Answering Test Script")
    print("Make sure the AI layer is running: python server_local.py")
    print()
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:3001/health', timeout=2)
        if response.status_code == 200:
            print("✅ AI layer is running")
            print()
            test_chat()
        else:
            print("❌ AI layer returned unexpected status")
    except:
        print("❌ AI layer is not running")
        print("Start it with: python server_local.py")
