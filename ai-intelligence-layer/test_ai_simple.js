// Test script for simplified AI router
const { analyzeImage } = require('./aiRouter_simple');

async function testAI() {
    console.log('🧪 Testing AI Intelligence Layer...\n');

    // Test 1: Low ML score (should skip AI analysis)
    console.log('Test 1: Low ML Score (should skip)');
    const result1 = await analyzeImage({
        imageData: 'dummy_base64_data',
        mlScore: 15,
        mlFactors: {},
        cameraId: 'TEST-CAM-001'
    });
    console.log('Result:', result1);
    console.log('Expected: provider="none", aiScore=15\n');

    // Test 2: High ML score with weapon (should use available provider)
    console.log('Test 2: High ML Score with Weapon');
    const result2 = await analyzeImage({
        imageData: 'dummy_base64_data',
        mlScore: 85,
        mlFactors: { weapon: 0.9, aggression: 0.7 },
        cameraId: 'TEST-CAM-002'
    });
    console.log('Result:', result2);
    console.log('Expected: High aiScore, sceneType="real_fight"\n');

    // Test 3: Medium ML score (should use available provider)
    console.log('Test 3: Medium ML Score');
    const result3 = await analyzeImage({
        imageData: 'dummy_base64_data',
        mlScore: 65,
        mlFactors: { aggression: 0.6, proximity: 0.5 },
        cameraId: 'TEST-CAM-003'
    });
    console.log('Result:', result3);
    console.log('Expected: Moderate aiScore\n');

    console.log('✅ All tests completed!');
    console.log('\nProvider used:', result2.provider);
    console.log('If provider is "fallback", consider installing Ollama for better results.');
    console.log('See AI_SETUP_GUIDE.md for instructions.');
}

testAI().catch(console.error);
