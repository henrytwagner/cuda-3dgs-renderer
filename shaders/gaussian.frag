#version 410 core

in vec2 vUv;

out vec4 fragColor;

uniform vec2 uMean;
uniform mat2 uCovInv;
uniform float uAmplitude;
uniform vec3 uColor;
uniform float uOpacity;

// Spherical harmonics uniforms
uniform vec3 uViewDir;           // Normalized view direction (world space)
uniform vec3 uShDc;              // DC coefficients (3 floats)
uniform float uShRest[45];       // Rest coefficients (15 per channel = 45 total)
uniform bool uSolidMode;         // If true, render solid (no opacity falloff)

// Evaluate spherical harmonics color for a given view direction
// Standard 3DGS uses 4 bands: DC (1) + rest (15) = 16 coefficients per channel
// Rest coefficients: l=1 (3) + l=2 (5) + l=3 (7) = 15 total
vec3 evalSHColor(vec3 viewDir, vec3 shDc, float shRest[45]) {
    // Normalize view direction
    vec3 dir = normalize(viewDir);
    float x = dir.x, y = dir.y, z = dir.z;
    
    // Precompute common terms for efficiency
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float x3 = x2 * x;
    float y3 = y2 * y;
    float z3 = z2 * z;
    
    // Evaluate SH basis functions for 4 bands (l=0,1,2,3)
    // Rest coefficients: 15 basis functions (indices 0-14)
    // Index mapping: 0-2 (l=1), 3-7 (l=2), 8-14 (l=3)
    float sh[15];
    
    // Band 1 (l=1): 3 coefficients
    sh[0] = 0.488603f * y;                               // Y_1^{-1}
    sh[1] = 0.488603f * z;                               // Y_1^0
    sh[2] = 0.488603f * x;                               // Y_1^1
    
    // Band 2 (l=2): 5 coefficients
    sh[3] = 1.092548f * x * y;                           // Y_2^{-2}
    sh[4] = 1.092548f * y * z;                           // Y_2^{-1}
    sh[5] = 0.315392f * (3.0f * z2 - 1.0f);             // Y_2^0
    sh[6] = 1.092548f * x * z;                           // Y_2^1
    sh[7] = 0.546274f * (x2 - y2);                       // Y_2^2
    
    // Band 3 (l=3): 7 coefficients
    sh[8] = 0.5900435899266435f * y * (3.0f * x2 - y2);  // Y_3^{-3}
    sh[9] = 2.890611442640554f * x * y * z;              // Y_3^{-2}
    sh[10] = 0.4570457994644658f * y * (5.0f * z2 - 1.0f); // Y_3^{-1}
    sh[11] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f); // Y_3^0
    sh[12] = 0.4570457994644658f * x * (5.0f * z2 - 1.0f); // Y_3^1
    sh[13] = 1.445305721320277f * z * (x2 - y2);          // Y_3^2
    sh[14] = 0.5900435899266435f * x * (x2 - 3.0f * y2);  // Y_3^3
    
    vec3 color = vec3(0.0);
    
    // For each RGB channel
    for (int c = 0; c < 3; ++c) {
        // Start with DC coefficient
        float result = shDc[c];
        
        // Add contributions from all 15 rest coefficients
        // shRest layout: [R_rest_0..14, G_rest_0..14, B_rest_0..14]
        for (int i = 0; i < 15; ++i) {
            int restIdx = c * 15 + i; // Index into shRest array
            result += shRest[restIdx] * sh[i];
        }
        
        // Apply sigmoid activation: C = 0.5 + 0.5 * tanh(SH_result)
        // This ensures color stays in [0, 1] range
        color[c] = 0.5f + 0.5f * tanh(result);
    }
    
    return color;
}

void main() {
    vec2 diff = vUv - uMean;
    float exponent = -0.5 * dot(diff, uCovInv * diff);
    float value = uAmplitude * exp(exponent);

    // Evaluate view-dependent color using spherical harmonics
    vec3 color = evalSHColor(uViewDir, uShDc, uShRest);
    
    if (uSolidMode) {
        // Solid mode: constant opacity, no falloff
        // Use a step function to create a hard edge at the Gaussian boundary
        // This creates a solid ellipse/circle shape
        float threshold = 0.1; // Adjust this to change the size of the solid shape
        if (value > threshold) {
            fragColor = vec4(color * uOpacity, uOpacity);
        } else {
            discard; // Don't render outside the Gaussian
        }
    } else {
        // Normal mode: opacity falls off with Gaussian value
        float alpha = clamp(value * uOpacity, 0.0, 1.0);
        vec3 rgb = color * value * uOpacity;
        fragColor = vec4(rgb, alpha);
    }
}