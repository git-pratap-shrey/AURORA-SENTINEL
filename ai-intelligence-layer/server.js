// ai-intelligence-layer/server.js
const express = require('express');
const bodyParser = require('body-parser');
// Use simplified router that works with free providers
const { analyzeImage } = require('./aiRouter_simple');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3001;

// Large limit for base64 images
app.use(bodyParser.json({ limit: '50mb' }));

app.post('/analyze', async (req, res) => {
    try {
        const { imageData, mlScore, mlFactors, cameraId, modelOverride } = req.body;

        if (!imageData) {
            return res.status(400).json({ error: 'imageData is required' });
        }

        console.log(`[AI-LAYER] Analyzing frame from ${cameraId} (ML Score: ${mlScore}%)`);
        const result = await analyzeImage({ imageData, mlScore, mlFactors, cameraId, modelOverride });

        console.log(`[AI-LAYER] Result from ${result.provider}: ${result.explanation.substring(0, 50)}...`);
        res.json(result);
    } catch (error) {
        console.error('[AI-LAYER] Error:', error);
        res.status(500).json({ error: 'Internal AI processing error', detail: error.message });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'ai-intelligence-layer' });
});

app.listen(port, () => {
    console.log(`AI Intelligence Layer listening at http://localhost:${port}`);
});
