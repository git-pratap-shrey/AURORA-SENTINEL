"""
AI Intelligence Layer - Local Server with HuggingFace Transformers
Flask server that uses local models (no API calls needed)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from aiRouter_enhanced import analyze_image, answer_question

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'ai-intelligence-layer-local',
        'models': 'transformers-local'
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze image endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'imageData' not in data:
            return jsonify({'error': 'imageData is required'}), 400
        
        image_data = data.get('imageData')
        ml_score = data.get('mlScore', data.get('riskScore', 50))
        ml_factors = data.get('mlFactors', {})
        camera_id = data.get('cameraId', 'UNKNOWN')
        
        logger.info(f"[AI-LAYER] Analyzing frame from {camera_id} (ML Score: {ml_score}%)")
        
        # Analyze image
        result = analyze_image(image_data, ml_score, ml_factors, camera_id)
        
        logger.info(f"[AI-LAYER] Result from {result['provider']}: {result['explanation'][:50]}...")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[AI-LAYER] Error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal AI processing error',
            'detail': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat/Question-answering endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'imageData' not in data or 'question' not in data:
            return jsonify({'error': 'imageData and question are required'}), 400
        
        image_data = data.get('imageData')
        question = data.get('question')
        
        logger.info(f"[AI-LAYER] Chat question: {question}")
        
        # Answer question
        result = answer_question(image_data, question)
        
        logger.info(f"[AI-LAYER] Answer from {result['provider']}: {result['answer'][:50]}...")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"[AI-LAYER] Chat error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal chat processing error',
            'detail': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3001))
    logger.info(f"Starting AI Intelligence Layer (Local Models) on port {port}")
    logger.info("Using HuggingFace Transformers with local models")
    logger.info("No API keys required - all processing is local")
    
    app.run(host='0.0.0.0', port=port, debug=False)
