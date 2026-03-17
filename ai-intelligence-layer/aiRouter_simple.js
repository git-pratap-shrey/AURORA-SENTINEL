// ai-intelligence-layer/aiRouter_simple.js
// Simplified AI Router without Genkit - Direct API calls to free providers
require('dotenv').config();
const axios = require('axios');

/**
 * Analyze image using available AI providers
 * Priority: Ollama (local, free) -> HuggingFace (cloud, free) -> Fallback
 */
async function analyzeImage({ imageData, mlScore, mlFactors, cameraId, modelOverride }) {
    // AI ALWAYS runs - no skipping based on ML score
    const prompt = `System: Aurora Sentinel AI Verification Expert.

Context:
- Camera: ${cameraId}
- ML Detection Score: ${mlScore}%
- ML Factors: ${JSON.stringify(mlFactors || {})}

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

    // Try providers in order: Ollama -> HuggingFace -> Fallback
    
    // 1. Try Ollama (Local, Free, No API Key Required)
    try {
        console.log('[AI-LAYER] Trying Ollama (local)...');
        const result = await tryOllama(imageData, prompt);
        if (result) {
            console.log('[AI-LAYER] ✅ Ollama succeeded');
            return result;
        }
    } catch (error) {
        console.log('[AI-LAYER] ⚠️ Ollama failed:', error.message);
    }

    // 2. Try HuggingFace (Cloud, Free with API Key)
    if (process.env.HF_ACCESS_TOKEN) {
        try {
            console.log('[AI-LAYER] Trying HuggingFace...');
            const result = await tryHuggingFace(imageData, prompt);
            if (result) {
                console.log('[AI-LAYER] ✅ HuggingFace succeeded');
                return result;
            }
        } catch (error) {
            console.log('[AI-LAYER] ⚠️ HuggingFace failed:', error.message);
        }
    } else {
        console.log('[AI-LAYER] ⚠️ HuggingFace skipped (no API key)');
    }

    // 3. Fallback: Rule-based analysis
    console.log('[AI-LAYER] Using fallback rule-based analysis');
    return fallbackAnalysis(mlScore, mlFactors);
}

/**
 * Try Ollama local API with enhanced error handling
 */
async function tryOllama(imageData, prompt) {
    const ollamaBase = process.env.OLLAMA_API_BASE || 'http://127.0.0.1:11434';
    const maxRetries = 2;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            if (attempt > 0) {
                console.log(`[AI-LAYER] Ollama retry ${attempt}/${maxRetries}`);
                await sleep(1000 * attempt); // Exponential backoff
            }
            
            // Check if Ollama is running and has models
            const tagsResponse = await axios.get(`${ollamaBase}/api/tags`, { 
                timeout: 3000,
                validateStatus: (status) => status < 500
            });
            
            if (tagsResponse.status !== 200) {
                throw new Error(`Ollama API returned status ${tagsResponse.status}`);
            }
            
            // Check if llava model is available
            const models = tagsResponse.data?.models || [];
            const hasLlava = models.some(m => m.name && m.name.includes('llava'));
            
            if (!hasLlava) {
                throw new Error('llava model not found. Run: ollama pull llava');
            }
            
            // Prepare image data
            const base64Image = imageData.replace(/^data:image\/\w+;base64,/, '');
            
            // Validate image data
            if (!base64Image || base64Image.length < 100) {
                throw new Error('Invalid image data for Ollama');
            }
            
            // Try to use llava model for vision
            const response = await axios.post(`${ollamaBase}/api/generate`, {
                model: 'llava',
                prompt: prompt,
                images: [base64Image],
                stream: false,
                options: {
                    temperature: 0.7,
                    top_p: 0.9
                }
            }, { 
                timeout: 30000, // 30 seconds for vision models
                validateStatus: (status) => status < 500
            });

            if (response.status === 404) {
                throw new Error('llava model not found in Ollama');
            }
            
            if (response.status !== 200) {
                throw new Error(`Ollama returned status ${response.status}`);
            }

            if (response.data && response.data.response) {
                const aiResponse = parseAIResponse(response.data.response, 'ollama');
                console.log(`[AI-LAYER] Ollama response: ${response.data.response.substring(0, 100)}...`);
                return aiResponse;
            } else {
                throw new Error('Empty response from Ollama');
            }
        } catch (error) {
            // Handle specific error types
            if (error.code === 'ECONNREFUSED') {
                throw new Error('Ollama not running. Start with: ollama serve');
            }
            
            if (error.code === 'ETIMEDOUT' || error.message.includes('timeout')) {
                console.log(`[AI-LAYER] Ollama timeout${attempt < maxRetries ? ', retrying...' : ''}`);
                if (attempt < maxRetries) continue;
                throw new Error('Ollama timeout - model may be loading or system is slow');
            }
            
            if (error.response && error.response.status === 404) {
                throw new Error('llava model not found. Run: ollama pull llava');
            }
            
            if (error.message.includes('llava model not found')) {
                throw error; // Don't retry if model is missing
            }
            
            // For other errors, retry if attempts remain
            if (attempt < maxRetries) {
                console.log(`[AI-LAYER] Ollama error: ${error.message}, retrying...`);
                continue;
            }
            
            throw error;
        }
    }
    
    return null;
}

/**
 * Try HuggingFace Inference API with enhanced error handling and retry logic
 */
async function tryHuggingFace(imageData, prompt) {
    const apiKey = process.env.HF_ACCESS_TOKEN;
    if (!apiKey) return null;

    // Expanded list of models with different capabilities
    const models = [
        // Vision-Language Models (best for scene understanding)
        { name: 'Salesforce/blip-image-captioning-large', timeout: 30000, retries: 2 },
        { name: 'Salesforce/blip-image-captioning-base', timeout: 25000, retries: 2 },
        { name: 'nlpconnect/vit-gpt2-image-captioning', timeout: 25000, retries: 2 },
        
        // Alternative captioning models
        { name: 'microsoft/git-base-coco', timeout: 25000, retries: 1 },
        { name: 'ydshieh/vit-gpt2-coco-en', timeout: 20000, retries: 1 },
        
        // Fallback to simpler models
        { name: 'google/vit-base-patch16-224', timeout: 15000, retries: 1 }
    ];

    let lastError = null;

    for (const modelConfig of models) {
        const { name: model, timeout, retries } = modelConfig;
        
        // Try each model with retry logic
        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                if (attempt > 0) {
                    console.log(`[AI-LAYER] Retry ${attempt}/${retries} for model: ${model}`);
                    // Exponential backoff: wait before retry
                    await sleep(1000 * Math.pow(2, attempt - 1));
                } else {
                    console.log(`[AI-LAYER] Trying HuggingFace model: ${model}`);
                }
                
                // Prepare image data (remove data URL prefix if present)
                const base64Image = imageData.replace(/^data:image\/\w+;base64,/, '');
                
                // Validate base64 image
                if (!base64Image || base64Image.length < 100) {
                    throw new Error('Invalid image data');
                }
                
                // Convert base64 to binary for HuggingFace API
                const imageBuffer = Buffer.from(base64Image, 'base64');
                
                // Validate buffer size (should be reasonable for an image)
                if (imageBuffer.length < 1000 || imageBuffer.length > 10 * 1024 * 1024) {
                    throw new Error(`Invalid image size: ${imageBuffer.length} bytes`);
                }
                
                const response = await axios.post(
                    `https://api-inference.huggingface.co/models/${model}`,
                    imageBuffer,
                    {
                        headers: {
                            'Authorization': `Bearer ${apiKey}`,
                            'Content-Type': 'application/octet-stream'
                        },
                        timeout: timeout,
                        validateStatus: (status) => status < 500 // Don't throw on 4xx errors
                    }
                );

                // Handle different HTTP status codes
                if (response.status === 503) {
                    console.log(`[AI-LAYER] Model ${model} loading (503)${attempt < retries ? ', retrying...' : ', trying next model'}`);
                    if (attempt < retries) continue; // Retry same model
                    else break; // Try next model
                }
                
                if (response.status === 429) {
                    console.log(`[AI-LAYER] Rate limit exceeded (429)${attempt < retries ? ', retrying after delay...' : ', trying next model'}`);
                    if (attempt < retries) {
                        await sleep(5000); // Wait 5 seconds for rate limit
                        continue;
                    } else break;
                }
                
                if (response.status === 410) {
                    console.log(`[AI-LAYER] Model ${model} deprecated (410), trying next model`);
                    break; // No point retrying deprecated model
                }
                
                if (response.status === 401 || response.status === 403) {
                    console.log(`[AI-LAYER] Authentication failed (${response.status}), check HF_ACCESS_TOKEN`);
                    throw new Error('Invalid HuggingFace API token');
                }
                
                if (response.status >= 400) {
                    console.log(`[AI-LAYER] Model ${model} error (${response.status}): ${JSON.stringify(response.data)}`);
                    break; // Try next model
                }

                // Success! Parse response
                if (response.data) {
                    let caption = null;
                    
                    // Handle different response formats
                    if (Array.isArray(response.data) && response.data[0]) {
                        caption = response.data[0].generated_text || response.data[0].caption || response.data[0].label;
                    } else if (response.data.generated_text) {
                        caption = response.data.generated_text;
                    } else if (response.data.caption) {
                        caption = response.data.caption;
                    } else if (response.data.label) {
                        caption = response.data.label;
                    } else if (typeof response.data === 'string') {
                        caption = response.data;
                    } else if (response.data.error) {
                        console.log(`[AI-LAYER] Model ${model} returned error: ${response.data.error}`);
                        break; // Try next model
                    }
                    
                    if (caption && caption.length > 0) {
                        console.log(`[AI-LAYER] ✅ HuggingFace success with ${model}`);
                        console.log(`[AI-LAYER] Caption: ${caption}`);
                        return analyzeCaption(caption, prompt);
                    } else {
                        console.log(`[AI-LAYER] Model ${model} returned empty caption`);
                        break; // Try next model
                    }
                }
            } catch (error) {
                lastError = error;
                
                // Handle network errors
                if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
                    console.log(`[AI-LAYER] Network error: ${error.message}`);
                    throw new Error('Cannot reach HuggingFace API - check internet connection');
                }
                
                if (error.code === 'ETIMEDOUT' || error.message.includes('timeout')) {
                    console.log(`[AI-LAYER] Model ${model} timeout${attempt < retries ? ', retrying...' : ', trying next model'}`);
                    if (attempt < retries) continue; // Retry
                    else break; // Try next model
                }
                
                // Handle axios errors
                if (error.response) {
                    const status = error.response.status;
                    console.log(`[AI-LAYER] Model ${model} HTTP error ${status}`);
                    break; // Try next model
                }
                
                // Unknown error
                console.log(`[AI-LAYER] Model ${model} unexpected error: ${error.message}`);
                if (attempt < retries) continue; // Retry
                else break; // Try next model
            }
        }
    }
    
    // All models failed
    const errorMsg = lastError ? lastError.message : 'All HuggingFace models unavailable';
    throw new Error(`HuggingFace failed: ${errorMsg}`);
}

/**
 * Sleep helper for retry delays
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Analyze caption from image captioning model with enhanced intelligence
 */
function analyzeCaption(caption, prompt) {
    const lowerCaption = caption.toLowerCase();
    
    let aiScore = 50;
    let sceneType = 'normal';
    let explanation = `Image analysis: ${caption}`;
    let confidence = 0.6;

    // Enhanced keyword detection with context and weights
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

    // Calculate weighted scores for each category
    const scores = {
        realFight: 0,
        boxing: 0,
        normal: 0,
        drama: 0
    };

    // Count matches with weights
    for (const [category, levels] of Object.entries(keywords)) {
        levels.high.forEach(kw => {
            if (lowerCaption.includes(kw)) scores[category] += 3;
        });
        levels.medium.forEach(kw => {
            if (lowerCaption.includes(kw)) scores[category] += 2;
        });
        levels.low.forEach(kw => {
            if (lowerCaption.includes(kw)) scores[category] += 1;
        });
    }

    // Determine scene type based on highest score
    const maxScore = Math.max(...Object.values(scores));
    const dominantCategory = Object.keys(scores).find(k => scores[k] === maxScore);

    // Decision logic with nuanced scoring
    if (scores.realFight >= 5 && scores.boxing < 3) {
        // Strong fight indicators without boxing context
        sceneType = 'real_fight';
        aiScore = Math.min(80, 60 + (scores.realFight * 4));
        explanation = `Detected potential violence: ${caption}`;
        confidence = 0.8;
    } else if (scores.realFight >= 3 && scores.boxing === 0 && scores.drama === 0) {
        // Moderate fight indicators, no controlled context
        sceneType = 'real_fight';
        aiScore = Math.min(70, 50 + (scores.realFight * 5));
        explanation = `Detected fight-like activity: ${caption}`;
        confidence = 0.7;
    } else if (scores.boxing >= 3) {
        // Clear boxing/training context
        sceneType = 'boxing';
        aiScore = Math.max(13, Math.min(35, 15 + (scores.boxing * 3)));
        explanation = `Detected controlled activity: ${caption}`;
        confidence = 0.8;
    } else if (scores.drama >= 3) {
        // Drama/performance context
        sceneType = 'drama';
        aiScore = Math.max(10, Math.min(30, 12 + (scores.drama * 3)));
        explanation = `Detected staged performance: ${caption}`;
        confidence = 0.75;
    } else if (scores.realFight >= 2 && scores.boxing >= 2) {
        // Mixed signals - could be boxing or real fight
        sceneType = 'boxing';
        aiScore = Math.min(45, 25 + (scores.realFight * 3));
        explanation = `Ambiguous: possible controlled combat activity: ${caption}`;
        confidence = 0.6;
    } else if (scores.normal >= 3) {
        // Clear normal activity
        sceneType = 'normal';
        aiScore = Math.max(10, Math.min(25, 10 + (scores.normal * 2)));
        explanation = `Normal activity detected: ${caption}`;
        confidence = 0.7;
    } else if (scores.realFight >= 1) {
        // Weak fight signal
        sceneType = 'normal';
        aiScore = Math.min(50, 30 + (scores.realFight * 5));
        explanation = `Low-level activity detected: ${caption}`;
        confidence = 0.5;
    } else {
        // Unclear - use moderate score
        sceneType = 'normal';
        aiScore = 30;
        explanation = `Unclear scene: ${caption}`;
        confidence = 0.5;
    }

    // Cap scores at 100
    aiScore = Math.min(100, Math.max(0, aiScore));

    return {
        aiScore: Math.round(aiScore),
        sceneType,
        explanation,
        confidence: Math.round(confidence * 100) / 100,
        provider: 'huggingface',
        debug: { scores, dominantCategory } // For debugging
    };
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
        // Parse text response
        const lowerText = text.toLowerCase();
        
        if (lowerText.includes('boxing') || lowerText.includes('sparring') || lowerText.includes('training')) {
            sceneType = 'boxing';
            aiScore = 25;
            explanation = 'Detected controlled activity (boxing/training)';
        } else if (lowerText.includes('drama') || lowerText.includes('acting') || lowerText.includes('performance')) {
            sceneType = 'drama';
            aiScore = 20;
            explanation = 'Detected staged performance';
        } else if (lowerText.includes('fight') || lowerText.includes('violence') || lowerText.includes('assault')) {
            sceneType = 'real_fight';
            aiScore = 75;
            explanation = 'Detected potential real violence';
        } else {
            sceneType = 'normal';
            aiScore = 30;
            explanation = 'Normal activity detected';
        }
    }

    return {
        aiScore,
        sceneType,
        explanation,
        confidence,
        provider
    };
}

/**
 * Fallback rule-based analysis when no AI provider is available
 */
function fallbackAnalysis(mlScore, mlFactors) {
    console.log('[AI-LAYER] Using rule-based fallback analysis');
    
    // Ensure mlScore is a valid number
    mlScore = mlScore || 0;
    
    // Simple rule-based logic based on ML factors
    let aiScore = mlScore * 0.8;  // Default: slightly lower than ML
    let sceneType = 'normal';
    let explanation = 'Rule-based analysis (no AI provider available)';
    
    // Check ML factors for patterns (check actual values, not just keys)
    if (mlFactors) {
        const weaponScore = mlFactors.weapon_detection || 0;
        const aggressionScore = mlFactors.aggressive_posture || 0;
        const proximityScore = mlFactors.proximity_violation || 0;
        const grapplingScore = mlFactors.grappling || 0;
        
        if (weaponScore > 0.4) {
            sceneType = 'real_fight';
            aiScore = Math.max(mlScore, 80);
            explanation = 'Weapon detected - high threat';
        } else if (grapplingScore > 0.5 && proximityScore > 0.3) {
            sceneType = 'real_fight';
            aiScore = mlScore * 0.9;
            explanation = 'Close combat detected (grappling + proximity)';
        } else if (aggressionScore > 0.6 && proximityScore > 0.3) {
            sceneType = 'real_fight';
            aiScore = mlScore * 0.85;
            explanation = 'Aggressive behavior with proximity detected';
        } else if (aggressionScore > 0.4 || proximityScore > 0.3) {
            sceneType = 'normal';
            aiScore = mlScore * 0.6;
            explanation = 'Moderate activity detected';
        } else {
            sceneType = 'normal';
            aiScore = mlScore * 0.5;
            explanation = 'Low risk activity';
        }
    }

    return {
        aiScore: Math.round(aiScore) || 0,  // Ensure it's a number, default to 0
        sceneType,
        explanation,
        confidence: 0.5,
        provider: 'fallback'
    };
}

module.exports = { analyzeImage };
