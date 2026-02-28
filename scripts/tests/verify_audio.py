
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.services.audio_service import audio_service
    print("AudioService imported successfully.")
    
    # Check if model loads (lazy load test)
    # We won't actually load it to save time/memory in this quick check, 
    # unless we want to be 100% sure. 
    # Let's just check imports inside the module.
    import transformers
    import librosa
    import soundfile
    print("Dependencies (transformers, librosa, soundfile) imported successfully.")

except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("Audio Processing Setup Verification Passed.")
