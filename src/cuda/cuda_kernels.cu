#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// CUDA kernel implementations for Gaussian splat rendering
// Implements Algorithm 2: GPU software rasterization of 3D Gaussians

// Tile size (16x16 pixels as per proposal)
#define TILE_SIZE 16

// Helper: Evaluate 2D Gaussian
// Returns the Gaussian value at pixel (px, py) given mean (uv) and inverse covariance
__device__ float eval2DGaussian(float px, float py, float meanU, float meanV, 
                                float covInv00, float covInv01, float covInv11) {
    float dx = px - meanU;
    float dy = py - meanV;
    
    // Compute: -0.5 * diff^T * covInv * diff
    float term1 = covInv00 * dx + covInv01 * dy;
    float term2 = covInv01 * dx + covInv11 * dy;
    float exponent = -0.5f * (dx * term1 + dy * term2);
    
    return expf(exponent);
}

// Helper: Evaluate spherical harmonics (4 bands)
// Ported from GLSL shader
__device__ void evalSHColor(const float* viewDir, const float* shDc, const float* shRest,
                            float* color) {
    // Normalize view direction
    float x = viewDir[0], y = viewDir[1], z = viewDir[2];
    float len = sqrtf(x*x + y*y + z*z);
    if (len > 1e-6f) {
        x /= len; y /= len; z /= len;
    }
    
    // Precompute common terms
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float x3 = x2 * x;
    float y3 = y2 * y;
    float z3 = z2 * z;
    
    // Evaluate SH basis functions (4 bands: l=0,1,2,3)
    float sh[15];
    
    // Band 1 (l=1)
    sh[0] = 0.488603f * y;
    sh[1] = 0.488603f * z;
    sh[2] = 0.488603f * x;
    
    // Band 2 (l=2)
    sh[3] = 1.092548f * x * y;
    sh[4] = 1.092548f * y * z;
    sh[5] = 0.315392f * (3.0f * z2 - 1.0f);
    sh[6] = 1.092548f * x * z;
    sh[7] = 0.546274f * (x2 - y2);
    
    // Band 3 (l=3)
    sh[8] = 0.5900435899266435f * y * (3.0f * x2 - y2);
    sh[9] = 2.890611442640554f * x * y * z;
    sh[10] = 0.4570457994644658f * y * (5.0f * z2 - 1.0f);
    sh[11] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);
    sh[12] = 0.4570457994644658f * x * (5.0f * z2 - 1.0f);
    sh[13] = 1.445305721320277f * z * (x2 - y2);
    sh[14] = 0.5900435899266435f * x * (x2 - 3.0f * y2);
    
    // For each RGB channel
    for (int c = 0; c < 3; ++c) {
        float result = shDc[c];
        
        // Add contributions from rest coefficients (15 per channel)
        for (int i = 0; i < 15; ++i) {
            result += shRest[c * 15 + i] * sh[i];
        }
        
        // Apply sigmoid activation: C = 0.5 + 0.5 * tanh(SH_result)
        color[c] = 0.5f + 0.5f * tanhf(result);
    }
}

// Forward declaration
extern "C" {
// Algorithm 2, Step 7: BlendInOrder
// Per-tile blending kernel - renders all pixels in a tile
__global__ void blendTilesKernel(
    // Framebuffer (output)
    float4* framebuffer,
    int width, int height,
    
    // Screen-space Gaussians (M', S')
    const float2* meansScreen,      // UV coordinates (indexed by visible index)
    const float* covariancesScreen,  // 2x2 matrices (4 floats each, indexed by visible index)
    
    // Original Gaussian data (C, A) - indexed by original Gaussian index
    const float* opacities,
    const float* shDc,              // 3 floats per splat
    const float* shRest,           // 45 floats per splat
    const float* meansWorld,       // 3 floats per splat (for view direction)
    
    // Mapping: visible index → original index
    const int* visibleToOriginal,
    
    // Camera state (for view direction)
    float3 cameraPos,
    
    // Sorting data
    const int* sortedIndices,      // Sorted list L (indices into visible list)
    const int* tileRanges,         // Ranges R[tile] = [start, end]
    
    // Tile grid
    int numTilesX,
    int numVisibleGaussians
) {
    // Each thread block processes one tile
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tileIdx = tileY * numTilesX + tileX;
    
    // Each thread processes one pixel in the tile
    int pixelX = tileX * TILE_SIZE + threadIdx.x;
    int pixelY = tileY * TILE_SIZE + threadIdx.y;
    
    // Check if pixel is within image bounds
    if (pixelX >= width || pixelY >= height) {
        return;
    }
    
    // Convert pixel coordinates to UV [0, 1]
    float u = (pixelX + 0.5f) / width;
    float v = (pixelY + 0.5f) / height;
    
    // Get tile range
    int rangeStart = tileRanges[tileIdx * 2 + 0];
    int rangeEnd = tileRanges[tileIdx * 2 + 1];
    
    // Initialize pixel color (background)
    float4 pixelColor = make_float4(0.07f, 0.07f, 0.09f, 1.0f); // Dark background
    float accumulatedAlpha = 0.0f;
    
    // Early-out threshold
    const float EARLY_OUT_THRESHOLD = 0.99f;
    
    // Iterate through splats assigned to this tile (in depth order)
    if (rangeStart >= 0 && rangeEnd > rangeStart) {
        for (int i = rangeStart; i < rangeEnd && accumulatedAlpha < EARLY_OUT_THRESHOLD; ++i) {
            int splatIdx = sortedIndices[i];
            
            if (splatIdx < 0 || splatIdx >= numVisibleGaussians) continue;
            
            // Get screen-space mean (UV)
            float2 meanUV = meansScreen[splatIdx];
            
            // Get screen-space covariance (2x2 matrix)
            const float* cov = &covariancesScreen[splatIdx * 4];
            float cov00 = cov[0];
            float cov01 = cov[1];
            float cov10 = cov[2];
            float cov11 = cov[3];
            
            // Check if covariance is valid (determinant > threshold)
            float det = cov00 * cov11 - cov01 * cov10;
            if (det <= 1e-6f) continue;
            
            // Compute inverse covariance
            float invDet = 1.0f / det;
            float covInv00 = cov11 * invDet;
            float covInv01 = -cov01 * invDet;
            float covInv11 = cov00 * invDet;
            
            // Evaluate 2D Gaussian at this pixel
            float gaussianValue = eval2DGaussian(u, v, meanUV.x, meanUV.y, 
                                                  covInv00, covInv01, covInv11);
            
            // Map visible index to original Gaussian index
            int originalIdx = visibleToOriginal[splatIdx];
            if (originalIdx < 0) continue;
            
            // Get opacity (indexed by original Gaussian)
            float opacity = opacities[originalIdx];
            float alpha = fminf(gaussianValue * opacity, 1.0f);
            
            // Skip if alpha is too small
            if (alpha < 1e-4f) continue;
            
            // Compute view direction (from Gaussian center to camera)
            float3 meanWorld = make_float3(
                meansWorld[originalIdx * 3 + 0],
                meansWorld[originalIdx * 3 + 1],
                meansWorld[originalIdx * 3 + 2]
            );
            float3 viewDir = make_float3(
                cameraPos.x - meanWorld.x,
                cameraPos.y - meanWorld.y,
                cameraPos.z - meanWorld.z
            );
            float viewDirLen = sqrtf(viewDir.x*viewDir.x + viewDir.y*viewDir.y + viewDir.z*viewDir.z);
            if (viewDirLen > 1e-6f) {
                viewDir.x /= viewDirLen;
                viewDir.y /= viewDirLen;
                viewDir.z /= viewDirLen;
            }
            
            // Evaluate spherical harmonics for view-dependent color
            float viewDirArr[3] = {viewDir.x, viewDir.y, viewDir.z};
            const float* splatShDc = &shDc[originalIdx * 3];
            const float* splatShRest = &shRest[originalIdx * 45];
            float shColor[3];
            evalSHColor(viewDirArr, splatShDc, splatShRest, shColor);
            
            // Blend: front-to-back alpha compositing
            // C_out = C_out + (1 - alpha_out) * C_in * alpha_in
            // alpha_out = alpha_out + (1 - alpha_out) * alpha_in
            float oneMinusAlpha = 1.0f - accumulatedAlpha;
            pixelColor.x += oneMinusAlpha * shColor[0] * alpha;
            pixelColor.y += oneMinusAlpha * shColor[1] * alpha;
            pixelColor.z += oneMinusAlpha * shColor[2] * alpha;
            accumulatedAlpha += oneMinusAlpha * alpha;
        }
    }
    
    // Clamp final color
    pixelColor.x = fminf(fmaxf(pixelColor.x, 0.0f), 1.0f);
    pixelColor.y = fminf(fmaxf(pixelColor.y, 0.0f), 1.0f);
    pixelColor.z = fminf(fmaxf(pixelColor.z, 0.0f), 1.0f);
    pixelColor.w = 1.0f;
    
    // Write to framebuffer
    int pixelIdx = pixelY * width + pixelX;
    framebuffer[pixelIdx] = pixelColor;
}
} // extern "C"

