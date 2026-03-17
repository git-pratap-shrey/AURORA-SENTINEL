// ai-intelligence-layer/aiRouter_ollama_enhanced.js
// Enhanced AI Router with Multiple Ollama Models and Smart Selection
require('dotenv').config();
const axios = require('axios');

/**
 * Analyze image using Ollama with smart model selection
 * Falls back to HuggingFace API only if Ollama is unavailable
 */
async function analyzeImage({ imageData, mlScore, mlFactors, cameraId, modelOverride }) {
    // Validate inputs
    if (!imageData || typeof imageData !== 'string') {
        console.log('[AI-LAYER] ⚠️ Invalid image data, using fallback');
        return fallbackAnalysis(mlScore, mlFactors);
    }
    
    mlScore = typeof mlScore === 'number' ? mlScore : 0;
    mlFactors = mlFactors || {};
    cameraId = cameraId || 'unknown';
    
    const prompt = buildPrompt(cameraId, mlScore, mlFactors);
    const errors = [];
    
    // 1. Try Ollama with Smart Model Selection
    try {
        console.log('[AI-LAYER] Trying Ollama with smart model selection...');
        const result = await tryOllamaWithFallback(imageData, prompt, mlScore, modelOverride);
        if (result) {
            console.log(`[AI-LAYER] ✅ Ollama succeeded with model: ${result.model}`);
            return result;
        }
    } catch (error) {
        const errorMsg = error.message || 'Unknown error';
        console.log('[AI-LAYER] ⚠️ Ollama failed:', errorMsg);
        errors.push({ provider: 'ollama', error: errorMsg });
    }

    // 2. Fallback to HuggingFace API (only if Ollama unavailable)
    if (process.env.HF_ACCESS_TOKEN) {
        try {
            console.log('[AI-LAYER] Trying HuggingFace API fallback...');
            const result = await tryHuggingFace(imageData, prompt);
            if (result) {
                console.log('[AI-LAYER] ✅ HuggingFace succeeded');
                return result;
            }
        } catch (error) {
            const errorMsg = error.message || 'Unknown error';
            console.log('[AI-LAYER] ⚠️ HuggingFace failed:', errorMsg);
            errors.push({ provider: 'huggingface', error: errorMsg });
        }
    }

    // 3. Final Fallback: Rule-based analysis
    console.log('[AI-LAYER] Using rule-based fallback');
    const fallbackResult = fallbackAnalysis(mlScore, mlFactors);
    fallbackResult.errors = errors;
    return fallbackResult;
}

/**
 * Try Ollama with multiple models and smart selection
 */
async function tryOllamaWithFallback(imageData, prompt, mlScore, modelOverride) {
    const ollamaBase = process.env.OLLAMA_API_BASE || 'http://127.0.0.1:11434';
    
    // Check if Ollama is running
    try {
        const tagsResponse = await axios.get(`${ollamaBase}/api/tags`, { timeout: 3000 });
        if (tagsResponse.status !== 200) {
            throw new Error('Ollama not responding');
        }
        
        const availableModels = (tagsResponse.data?.models || []).map(m => m.name);
        console.log(`[AI-LAYER] Available Ollama models: ${availableModels.join(', ')}`);
        
        if (availableModels.length === 0) {
            throw new Error('No Ollama models installed');
        }
        
        // Smart model selection based on ML score
        let modelsToTry = [];
        
        if (modelOverride) {
            // User specified a model
            modelsToTry = [modelOverride];
        } else {
            // Smart selection based on ML score
            modelsToTry = selectModelsForScore(mlScore, availableModels);
        }
        
        // Try each model in order
        for (const model of modelsToTry) {
            try {
                console.log(`[AI-LAYER] Trying Ollama model: ${model}`);
                const result = await tryOllamaModel(ollamaBase, model, imageData, prompt);
                if (result) {
                    result.model = model;
                    return result;
                }
            } catch (error) {
                console.log(`[AI-LAYER] Model ${model} failed: ${error.message}`);
                continue;
            }
        }
        
        throw new Error('All Ollama models failed');
        
    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            throw new Error('Ollama not running. Start with: ollama serve');
        }
        throw error;
    }
}

/**
 * Select best models based on ML score
 */
function selectModelsForScore(mlScore, availableModels) {
    const modelPreferences = {
        high: ['llava:13b', 'llava', 'bakllava', 'llava-phi3', 'minicpm-v'],  // Best accuracy for high risk
        medium: ['llava', 'bakllava', 'llava:13b', 'llava-phi3', 'minicpm-v'], // Balanced
        low: ['llava-phi3', 'minicpm-v', 'llava', 'bakllava']                  // Fast for low risk
    };
    
    let preferred;
    if (mlScore > 70) {
        preferred = modelPreferences.high;
    } else if (mlScore > 50) {
        preferred = modelPreferences.medium;
    } else {
        preferred = modelPreferences.low;
    }
    
    // Filter to only available models, maintaining preference order
    const selected = preferred.filter(model => 
        availableModels.some(available => available.includes(model.split(':')[0]))
    );
    
    // Add any other available models as fallback
    availableModels.forEach(model => {
        if (!selected.some(s => model.includes(s.split(':')[0]))) {
            selected.push(model);
        }
    });
    
    return selected.slice(0, 3); // Try top 3 models
}

/**
 * Try a specific Ollama model
 */
async function tryOllamaModel(ollamaBase, model, imageData, prompt) {
    const base64Image = imageData.replace(/^data:image\/\w+;base64,/, '');
    
    if (!base64Image || base64Image.length < 100) {
        throw new Error('Invalid image data');
    }
    
    const response = await axios.post(`${ollamaBase}/api/generate`, {
        model: model,
        prompt: prompt,
        images: [base64Image],
        stream: false,
        options: {
            temperature: 0.7,
            top_p: 0.9,
            num_predict: 200  // Limit response length
        }
    }, { 
        timeout: 30000,
        validateStatus: (status) => status < 500
    });

    if (response.status === 404) {
        throw new Error(`Model ${model} not found`);
    }
    
    if (response.status !== 200) {
        throw new Error(`Ollama returned status ${response.status}`);
    }

    if (response.data && response.data.response) {
        const aiResponse = parseAIResponse(response.data.response, 'ollama');
        console.log(`[AI-LAYER] ${model} response: ${response.data.response.substring(0, 100)}...`);
        return aiResponse;
    }
    
    throw new Error('Empty response from Ollama');
}

/**
 * Build prompt for AI analysis
 */
function buildPrompt(cameraId, mlScore, mlFactors) {
    return `System: Aurora Sentinel AI Verification Expert.

Context:
- Camera: ${cameraId}
- ML Detection Score: ${mlScore}%
- ML Factors: ${JSON.stringify(mlFactors)}

The ML system detected combat-like behavior at ${mlScore}%. Your task is to verify if this is:
1. A real threat (actual fight, assault, violence)
2. Controlled activity (boxing, sparring, martial arts training)
3. Staged/drama (acting, performance, rehearsal)
4. False positive (normal activity misclassified)

Analyze the image and provide:
- AI Risk Score (0-100): Your assessment of actual threat level
- Scene Type: real_fight | boxing | drama | normal
- Explanation: Brief reasoning for your assessment
- Confidence: How certain are you? (0.0-1.0)

Be objective. If ML detected combat poses but this is clearly controlled (boxing ring, protective gear, training environment), your AI score should be LOW. If this appears to be genuine violence, your AI score should be HIGH.

Respond in JSON format:
{
  "aiScore": <number 0-100>,
  "sceneType": "<real_fight|boxing|drama|normal>",
  "explanation": "<brief reasoning>",
  "confidence": <number 0.0-1.0>
}`;
}

/**
 * Parse AI response (JSON or text)
 */
function parseAIResponse(text, provider) {
    let aiScore = 50;
    let sceneType = 'normal';
    let explanation = text;
    let confidence = 0.5;

    try {
        // Try to parse JSON
        const parsed = JSON.parse(text);
        aiScore = parsed.aiScore || 50;
        sceneType = parsed.sceneType || 'normal';
        explanation = parsed.explanation || text;
        confidence = parsed.confidence || 0.5;
    } catch (e) {
        // Parse text response with enhanced keyword analysis
        const lowerText = text.toLowerCase();
        
        // Use the same enhanced caption analysis
        const result = analyzeTextResponse(text);
        aiScore = result.aiScore;
        sceneType = result.sceneType;
        explanation = result.explanation;
        confidence = result.confidence;
    }

    return {
        aiScore: Math.round(aiScore),
        sceneType,
        explanation,
        confidence: Math.round(confidence * 100) / 100,
        provider
    };
}

/**
 * Analyze text response with enhanced intelligence
 */
function analyzeTextResponse(text) {
    const lowerText = text.toLowerCase();
    
    const keywords = {
        realFight: {
            high: ['fight', 'fighting', 'violence', 'violent', 'assault', 'attack', 'attacking', 'brawl'],
            medium: ['punch', 'punching', 'kick', 'kicking', 'hitting', 'striking', 'combat', 'aggressive'],
            low: ['confrontation', 'argument', 'pushing', 'shoving', 'grabbing', 'wrestling']
        },
        boxing: {
            high: ['boxing', 'boxer', 'ring', 'gloves', 'sparring'],
            medium: ['training', 'gym', 'practice', 'exercise', 'workout', 'sport'],
            low: ['protective', 'gear', 'coach', 'referee']
        },
        normal: {
            high: ['standing', 'walking', 'talking', 'sitting', 'waiting'],
            medium: ['people', 'person', 'man', 'woman', 'group', 'crowd'],
            low: ['indoor', 'outdoor', 'hallway', 'corridor', 'room']
        },
        drama: {
            high: ['acting', 'performance', 'stage', 'theater', 'movie'],
            medium: ['scene', 'rehearsal', 'filming', 'camera'],
            low: ['costume', 'set', 'production']
        }
    };

    const scores = { realFight: 0, boxing: 0, normal: 0, drama: 0 };

    for (const [category, levels] of Object.entries(keywords)) {
        levels.high.forEach(kw => { if (lowerText.includes(kw)) scores[category] += 3; });
        levels.medium.forEach(kw => { if (lowerText.includes(kw)) scores[category] += 2; });
        levels.low.forEach(kw => { if (lowerText.includes(kw)) scores[category] += 1; });
    }

    let aiScore, sceneType, explanation, confidence;

    if (scores.realFight >= 5 && scores.boxing < 3) {
        sceneType = 'real_fight';
        aiScore = Math.min(80, 60 + (scores.realFight * 4));
        explanation = `Detected potential violence: ${text.substring(0, 100)}`;
        confidence = 0.8;
    } else if (scores.boxing >= 3) {
        sceneType = 'boxing';
        aiScore = Math.max(13, Math.min(35, 15 + (scores.boxing * 3)));
        explanation = `Detected controlled activity: ${text.substring(0, 100)}`;
        confidence = 0.8;
    } else if (scores.drama >= 3) {
        sceneType = 'drama';
        aiScore = Math.max(10, Math.min(30, 12 + (scores.drama * 3)));
        explanation = `Detected staged performance: ${text.substring(0, 100)}`;
        confidence = 0.75;
    } else if (scores.normal >= 3) {
        sceneType = 'normal';
        aiScore = Math.max(10, Math.min(25, 10 + (scores.normal * 2)));
        explanation = `Normal activity detected: ${text.substring(0, 100)}`;
        confidence = 0.7;
    } else {
        sceneType = 'normal';
        aiScore = 30;
        explanation = `Unclear scene: ${text.substring(0, 100)}`;
        confidence = 0.5;
    }

    return { aiScore, sceneType, explanation, confidence };
}

/**
 * HuggingFace fallback (simplified - only used if Ollama unavailable)
 */
async function tryHuggingFace(imageData, prompt) {
    // Simplified version - just try one model quickly
    const apiKey = process.env.HF_ACCESS_TOKEN;
    if (!apiKey) return null;

    const model = 'Salesforce/blip-image-captioning-base';
    const base64Image = imageData.replace(/^data:image\/\w+;base64,/, '');
    const imageBuffer = Buffer.from(base64Image, 'base64');
    
    const response = await axios.post(
        `https://api-inference.huggingface.co/models/${model}`,
        imageBuffer,
        {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/octet-stream'
            },
            timeout: 15000
        }
    );

    if (response.data && response.data[0] && response.data[0].generated_text) {
        const caption = response.data[0].generated_text;
        const result = analyzeTextResponse(caption);
        result.provider = 'huggingface';
        return result;
    }
    
    return null;
}

/**
 * Fallback rule-based analysis
 */
function fallbackAnalysis(mlScore, mlFactors) {
    mlScore = mlScore || 0;
    let aiScore = mlScore * 0.8;
    let sceneType = 'normal';
    let explanation = 'Rule-based analysis (no AI provider available)';
    
    if (mlFactors) {
        const weaponScore = mlFactors.weapon_detection || 0;
        const aggressionScore = mlFactors.aggressive_posture || 0;
        const proximityScore = mlFactors.proximity_violation || 0;
        
        if (weaponScore > 0.4) {
            sceneType = 'real_fight';
            aiScore = Math.max(mlScore, 80);
            explanation = 'Weapon detected - high threat';
        } else if (aggressionScore > 0.6 && proximityScore > 0.3) {
            sceneType = 'real_fight';
            aiScore = mlScore * 0.85;
            explanation = 'Aggressive behavior with proximity detected';
        } else {
            aiScore = mlScore * 0.6;
        }
    }

    return {
        aiScore: Math.round(aiScore) || 0,
        sceneType,
        explanation,
        confidence: 0.5,
        provider: 'fallback'
    };
}

module.exports = { analyzeImage };
