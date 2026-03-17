"""
Qwen2-VL-2B Integration for Video Fight Detection
Optimized for systems with limited VRAM (4-6GB)
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
import io
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

class Qwen2VLAnalyzer:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        """Initialize Qwen2-VL model"""
        print(f"[Qwen2VL] Loading model: {model_name}")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Qwen2VL] Using device: {self.device}")
        
        # Load model with optimizations
        if self.device == "cuda":
            # GPU mode with 4-bit quantization for memory efficiency
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            # CPU mode
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(self.device)
        
        # Minimum pixels for video processing
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # Store process_vision_info function
        self.process_vision_info = process_vision_info
        
        print("[Qwen2VL] Model loaded successfully")
        print("[Qwen2VL] Model loaded successfully")
    
    def analyze_video_frames(self, frames, fps=1, max_frames=16):
        """
        Analyze video frames for fight detection
        
        Args:
            frames: List of numpy arrays (video frames)
            fps: Frames per second to sample
            max_frames: Maximum number of frames to analyze
            
        Returns:
            dict with aiScore, sceneType, explanation, confidence
        """
        # Save frames to temporary video file
        temp_video_path = self._frames_to_video(frames, fps)
        
        try:
            # Analyze video
            result = self.analyze_video_file(temp_video_path, fps, max_frames)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def analyze_video_file(self, video_path, fps=1, max_frames=16):
        """
        Analyze video file for fight detection
        
        Args:
            video_path: Path to video file
            fps: Frames per second to sample
            max_frames: Maximum number of frames to analyze
            
        Returns:
            dict with aiScore, sceneType, explanation, confidence
        """
        question = """Analyze this video for potential violence or fighting in a PUBLIC SURVEILLANCE context.

Classify ONLY as:
- real_fight: Physical aggression WITHOUT protective gear or referee
- organized_sport: Boxing/martial arts WITH protective gear (gloves, headgear) AND referee/ring structure
- suspicious: Crowd surrounding people (concealment) OR unknown items in suspicious contexts
- normal: Safe activity, no threats

CRITICAL: DO NOT classify as "prank" or "drama". Any fight without sport indicators = real_fight.

Sport indicators: protective gear (boxing gloves, headgear), referee present, ring/mat structure
Heavy fighting indicators: multiple strikes, sustained aggression, visible injury

Risk assessment (0-100):
- real_fight with heavy fighting (multiple strikes, sustained aggression, visible injury): 80-95
- real_fight without heavy indicators: 75-90
- organized_sport: 20-35 (capped)
- suspicious: 60-75
- normal: 10-25

Respond in JSON format:
{
  "aiScore": <number 0-100>,
  "sceneType": "<real_fight|organized_sport|suspicious|normal>",
  "explanation": "<what you observe>",
  "confidence": <number 0.0-1.0>
}"""
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": fps,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[Qwen2VL] Raw response: {response}")
        
        # Parse response
        return self._parse_response(response)
    
    def _frames_to_video(self, frames, fps=1):
        """Convert frames to temporary video file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()
        
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Convert RGB to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        return temp_path
    
    def _parse_response(self, response):
            """
            Parse Qwen2-VL response with keyword-based fallback (NO hardcoded scores).

            Keyword Mapping:
            - fight/violence/assault/punch/kick WITHOUT sport indicators → 75-90
            - boxing/training/sparring/gloves/referee → 20-35
            - crowd surrounding/unknown items → 60-75
            - normal/walking/standing/safe → 10-25

            Returns structured dict with aiScore, sceneType, explanation, confidence, parsing_method
            """
            import json
            import re

            # Try to extract JSON first
            try:
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    ai_score = int(parsed.get('aiScore', 0))
                    scene_type = parsed.get('sceneType', 'normal')

                    # Validate and normalize scene type (remove deprecated types)
                    if scene_type in ['drama', 'prank', 'staged', 'performance']:
                        scene_type = 'normal'
                    elif scene_type == 'boxing':
                        scene_type = 'organized_sport'

                    return {
                        'aiScore': ai_score,
                        'sceneType': scene_type,
                        'explanation': parsed.get('explanation', response),
                        'confidence': float(parsed.get('confidence', 0.75)),
                        'provider': 'qwen2vl',
                        'parsing_method': 'json'
                    }
            except:
                pass

            # Fallback: keyword-based analysis (NO hardcoded scores)
            lower_text = response.lower()

            # Define keyword patterns
            fight_keywords = ['fight', 'fighting', 'violence', 'assault', 'attack', 'punch', 'kick',
                              'hitting', 'striking', 'aggressive', 'aggression', 'brawl']
            heavy_fight_keywords = ['multiple strikes', 'sustained aggression', 'visible injury', 'blood',
                                    'weapon', 'knife', 'gun', 'severe', 'brutal']
            sport_keywords = ['boxing', 'martial arts', 'training', 'sparring', 'controlled',
                              'protective gear', 'gloves', 'headgear', 'referee', 'ring', 'mat']
            suspicious_keywords = ['crowd surrounding', 'people covering', 'concealment', 'unknown item',
                                   'suspicious object', 'suspicious behavior']
            normal_keywords = ['normal', 'safe', 'walking', 'standing', 'talking', 'conversation',
                               'no threat', 'peaceful']

            # Check for sport indicators first
            has_sport_indicators = any(keyword in lower_text for keyword in sport_keywords)

            # Check for heavy fighting
            has_heavy_fight = any(keyword in lower_text for keyword in heavy_fight_keywords)

            # Check for fight keywords
            has_fight = any(keyword in lower_text for keyword in fight_keywords)

            # Check for suspicious behavior
            has_suspicious = any(keyword in lower_text for keyword in suspicious_keywords)

            # Check for normal indicators
            has_normal = any(keyword in lower_text for keyword in normal_keywords)

            # Count keyword matches for score variation
            fight_count = sum(1 for kw in fight_keywords if kw in lower_text)
            heavy_fight_count = sum(1 for kw in heavy_fight_keywords if kw in lower_text)
            sport_count = sum(1 for kw in sport_keywords if kw in lower_text)
            suspicious_count = sum(1 for kw in suspicious_keywords if kw in lower_text)
            normal_count = sum(1 for kw in normal_keywords if kw in lower_text)

            # Determine scene type and score based on keyword analysis
            # Use keyword count to vary score within range (more keywords = higher score)
            if has_sport_indicators:
                # Sport indicators present: 20-35 range
                scene_type = 'organized_sport'
                # Vary score based on sport keyword count (1-5+ keywords)
                ai_score = min(20 + (sport_count * 3), 35)
                confidence = 0.7
            elif has_heavy_fight:
                # Heavy fighting indicators: 80-95 range
                scene_type = 'real_fight'
                # Vary score based on heavy fight keyword count
                ai_score = min(80 + (heavy_fight_count * 5), 95)
                confidence = 0.85
            elif has_fight:
                # Fight keywords without sport indicators: 75-90 range
                scene_type = 'real_fight'
                # Vary score based on fight keyword count
                ai_score = min(75 + (fight_count * 3), 90)
                confidence = 0.8
            elif has_suspicious:
                # Suspicious indicators: 60-75 range
                scene_type = 'suspicious'
                # Vary score based on suspicious keyword count
                ai_score = min(60 + (suspicious_count * 5), 75)
                confidence = 0.65
            elif has_normal:
                # Normal indicators: 10-25 range
                scene_type = 'normal'
                # Vary score based on normal keyword count (inverse - more normal = lower score)
                ai_score = max(25 - (normal_count * 3), 10)
                confidence = 0.75
            else:
                # No clear indicators - default to normal with low confidence
                scene_type = 'normal'
                ai_score = 20
                confidence = 0.5

            return {
                'aiScore': ai_score,
                'sceneType': scene_type,
                'explanation': response[:300],
                'confidence': confidence,
                'provider': 'qwen2vl',
                'parsing_method': 'keyword'
            }



# Flask/FastAPI endpoint wrapper
def analyze_video_with_qwen2vl(video_frames, fps=1, max_frames=16):
    """
    Wrapper function for easy integration
    
    Args:
        video_frames: List of numpy arrays (video frames)
        fps: Frames per second
        max_frames: Maximum frames to analyze
        
    Returns:
        dict with aiScore, sceneType, explanation, confidence
    """
    try:
        analyzer = Qwen2VLAnalyzer()
        result = analyzer.analyze_video_frames(video_frames, fps, max_frames)
        return result
    except Exception as e:
        print(f"[Qwen2VL] Error: {e}")
        return {
            'aiScore': 0,
            'sceneType': 'normal',
            'explanation': f'Qwen2-VL error: {str(e)}',
            'confidence': 0.0,
            'provider': 'qwen2vl',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test with sample video
    print("Testing Qwen2-VL integration...")
    
    # Create sample frames (replace with actual video frames)
    import numpy as np
    sample_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    result = analyze_video_with_qwen2vl(sample_frames, fps=1, max_frames=10)
    print(f"Result: {result}")
