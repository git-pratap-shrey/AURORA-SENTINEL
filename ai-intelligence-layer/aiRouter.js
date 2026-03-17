// ai-intelligence-layer/aiRouter.js
require('dotenv').config();
const { genkit } = require('genkit');
const { googleAI, gemini15Flash } = require('@genkit-ai/googleai');
const { huggingface } = require('@genkit-ai/huggingface');
const { ollama } = require('@genkit-ai/ollama');
const { z } = require('zod');

// Configure Genkit with security in mind
const ai = genkit({
    plugins: [
        googleAI({ apiKey: process.env.GOOGLE_GENAI_API_KEY }),
        huggingface({ apiKey: process.env.HF_ACCESS_TOKEN }),
        ollama({ apiBase: process.env.OLLAMA_API_BASE || 'http://127.0.0.1:11434' }),
    ],
});

// The Intelligent Router Flow - ENHANCED FOR TWO-TIER SCORING
const analyzeImage = ai.defineFlow(
    {
        name: 'analyzeImage',
        inputSchema: z.object({
            imageData: z.string(), // base64
            mlScore: z.number(),   // ML risk score (0-100) - renamed from riskScore
            mlFactors: z.object({}).optional(), // ML detection factors
            cameraId: z.string(),
            timestamp: z.number().optional(),
            modelOverride: z.string().optional()
        }),
        outputSchema: z.object({
            aiScore: z.number(),         // AI risk score (0-100)
            explanation: z.string(),     // Context reasoning
            sceneType: z.string(),       // "real_fight" | "boxing" | "drama" | "normal"
            confidence: z.number(),      // AI confidence (0-1)
            provider: z.string()
        })
    },
    async (input) => {
        const { imageData, mlScore, mlFactors, cameraId, modelOverride } = input;

        // CREDIT SAFETY: Skip analysis if ML score is too low
        if (mlScore < 20 && !modelOverride) {
            return {
                aiScore: mlScore,
                explanation: "Static/Low-risk scene. No deep analysis triggered.",
                sceneType: "normal",
                confidence: 0.9,
                provider: "none"
            };
        }

        let result;
        let providerUsed = '';

        // Enhanced prompt for two-tier scoring
        const enhancedPrompt = `System: Aurora Sentinel AI Verification Expert.

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

        // ROUTING LOGIC
        // 1. Gemini (Premium) - Highest Accuracy, Highest Cost
        if (mlScore > 85 || modelOverride === 'gemini') {
            providerUsed = 'gemini';
            const { text } = await ai.generate({
                model: gemini15Flash,
                prompt: enhancedPrompt,
            });
            result = text;
        }
        // 2. Hugging Face (Cloud) - Balanced
        else if (mlScore > 50 || modelOverride === 'huggingface') {
            providerUsed = 'huggingface';
            const { text } = await ai.generate({
                model: 'huggingface/llava-hf/llava-1.5-7b-hf',
                prompt: enhancedPrompt,
            });
            result = text;
        }
        // 3. Ollama (Local) - Free
        else {
            providerUsed = 'ollama';
            const { text } = await ai.generate({
                model: 'ollama/llava',
                prompt: enhancedPrompt,
            });
            result = text;
        }

        // Parse AI response
        let aiScore = mlScore;  // Default to ML score
        let sceneType = "normal";
        let explanation = result;
        let confidence = 0.5;

        try {
            // Try to parse JSON response
            const parsed = JSON.parse(result);
            aiScore = parsed.aiScore || mlScore;
            sceneType = parsed.sceneType || "normal";
            explanation = parsed.explanation || result;
            confidence = parsed.confidence || 0.5;
        } catch (e) {
            // If not JSON, try to extract information from text
            // Look for keywords to determine scene type
            const lowerResult = result.toLowerCase();
            if (lowerResult.includes('boxing') || lowerResult.includes('sparring') || lowerResult.includes('training')) {
                sceneType = 'boxing';
                aiScore = Math.min(mlScore * 0.5, 30);  // Reduce score for controlled activity
            } else if (lowerResult.includes('drama') || lowerResult.includes('acting') || lowerResult.includes('performance')) {
                sceneType = 'drama';
                aiScore = Math.min(mlScore * 0.3, 20);
            } else if (lowerResult.includes('fight') || lowerResult.includes('violence') || lowerResult.includes('assault')) {
                sceneType = 'real_fight';
                aiScore = Math.max(mlScore, 70);  // Escalate for real threats
            }
        }

        return {
            aiScore: aiScore,
            explanation: explanation,
            sceneType: sceneType,
            confidence: confidence,
            provider: providerUsed
        };
    }
);

module.exports = { analyzeImage };
