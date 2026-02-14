import ollama
from PIL import Image
import io

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')

print("Testing PIL Image input...")
try:
    ollama.chat(model='llava', messages=[{'role': 'user', 'content': 'describe this', 'images': [img]}])
    print("SUCCESS: PIL Image accepted")
except Exception as e:
    print(f"FAILURE: PIL Image not accepted: {e}")

print("\nTesting Bytes input...")
try:
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    ollama.chat(model='llava', messages=[{'role': 'user', 'content': 'describe this', 'images': [img_bytes]}])
    print("SUCCESS: Bytes accepted")
except Exception as e:
    print(f"FAILURE: Bytes not accepted: {e}")
