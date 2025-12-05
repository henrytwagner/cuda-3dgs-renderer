#include <glad/glad.h>  // Must be included before cuda_gl_interop.h
#include <chrono>
#include <iomanip>
#include "cuda_renderer.h"
#include "benchmark.h"
#include "../utils/math_utils.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <tuple>

// Thrust for GPU sorting and parallel primitives
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

// Global benchmark instance
BenchmarkTracker g_benchmark;

using namespace glm;

// Tile size constant
constexpr int TILE_SIZE = 16;

// Batch size for shared memory tiling in blend kernel
constexpr int BLEND_BATCH_SIZE = 256;

// Constant memory for tile/screen dimensions (used by GPU kernels)
__constant__ int d_numTilesX;
__constant__ int d_numTilesY;
__constant__ int d_width;
__constant__ int d_height;

// Helper: Evaluate 2D Gaussian
__device__ float eval2DGaussian(float px, float py, float meanU, float meanV, 
                                float covInv00, float covInv01, float covInv11) {
    float dx = px - meanU;
    float dy = py - meanV;
    float term1 = covInv00 * dx + covInv01 * dy;
    float term2 = covInv01 * dx + covInv11 * dy;
    float exponent = -0.5f * (dx * term1 + dy * term2);
    return expf(exponent);
}

// Helper: Evaluate spherical harmonics (simplified - just DC term for now)
__device__ void evalSHColorSimple(const float* shDc, float* color) {
    // Just use DC term with sigmoid activation
    for (int c = 0; c < 3; ++c) {
        color[c] = 0.5f + 0.5f * tanhf(shDc[c]);
    }
}

__global__ void blendTilesKernel(
    float4* framebuffer, int width, int height,
    const float2* meansScreen, const float* covariancesScreen,
    const float* opacities, const float* shDc, const float* shRest,
    const float* meansWorld, const int* visibleToOriginal,
    float3 cameraPos,
    const int* sortedIndices, const int* tileRanges,
    int numTilesX, int numVisibleGaussians
) {
    // Shared memory for cooperative Gaussian loading
    __shared__ float2 s_means[BLEND_BATCH_SIZE];
    __shared__ float3 s_covInv[BLEND_BATCH_SIZE];
    __shared__ float4 s_colorOpacity[BLEND_BATCH_SIZE];
    
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tileIdx = tileY * numTilesX + tileX;
    
    int pixelX = tileX * TILE_SIZE + threadIdx.x;
    int pixelY = tileY * TILE_SIZE + threadIdx.y;
    int threadIdxLinear = threadIdx.y * TILE_SIZE + threadIdx.x;
    
    // Check if this pixel is valid (within screen bounds)
    bool pixelValid = (pixelX < width && pixelY < height);
    
    int rangeStart = tileRanges[tileIdx * 2 + 0];
    int rangeEnd = tileRanges[tileIdx * 2 + 1];
    
    float4 pixelColor = make_float4(0.07f, 0.07f, 0.09f, 1.0f);
    float accumulatedAlpha = 0.0f;
    
    // Handle empty tile - ALL threads take this path together
    if (rangeStart < 0 || rangeEnd <= rangeStart) {
        if (pixelValid) {
            int flippedX = width - 1 - pixelX;
            int flippedY = height - 1 - pixelY;
            framebuffer[flippedY * width + flippedX] = pixelColor;
        }
        return;
    }
    
    const float SH_C0 = 0.28209479177387814f;
    
    // Pixel position in NDC space
    float ndcX = ((pixelX + 0.5f) / width) * 2.0f - 1.0f;
    float ndcY = ((pixelY + 0.5f) / height) * 2.0f - 1.0f;
    
    int totalGaussians = rangeEnd - rangeStart;
    int numBatches = (totalGaussians + BLEND_BATCH_SIZE - 1) / BLEND_BATCH_SIZE;
    
    // Process Gaussians in batches
    for (int batch = 0; batch < numBatches; ++batch) {
        int batchStart = rangeStart + batch * BLEND_BATCH_SIZE;
        int batchEnd = min(batchStart + BLEND_BATCH_SIZE, rangeEnd);
        int batchSize = batchEnd - batchStart;
        
        // Cooperative load into shared memory
        if (threadIdxLinear < batchSize) {
            int globalIdx = batchStart + threadIdxLinear;
            int splatIdx = sortedIndices[globalIdx];
            
            // Load and transform mean to NDC
            float2 meanUV = meansScreen[splatIdx];
            s_means[threadIdxLinear] = make_float2(
                meanUV.x * 2.0f - 1.0f,
                meanUV.y * 2.0f - 1.0f
            );
            
            // Load covariance and pre-compute inverse
            float cov_a = covariancesScreen[splatIdx * 4 + 0];
            float cov_b = covariancesScreen[splatIdx * 4 + 1];
            float cov_c = covariancesScreen[splatIdx * 4 + 3];
            
            float det = cov_a * cov_c - cov_b * cov_b;
            if (det > 1e-10f) {
                float invDet = 1.0f / det;
                s_covInv[threadIdxLinear] = make_float3(
                    cov_c * invDet, -cov_b * invDet, cov_a * invDet
                );
            } else {
                s_covInv[threadIdxLinear] = make_float3(0.0f, 0.0f, 0.0f);
            }
            
            // Load color and opacity
            int originalIdx = visibleToOriginal[splatIdx];
            if (originalIdx >= 0) {
                float opacity = opacities[originalIdx];
                const float* dc = &shDc[originalIdx * 3];
                s_colorOpacity[threadIdxLinear] = make_float4(
                    fminf(fmaxf(SH_C0 * dc[0] + 0.5f, 0.0f), 1.0f),
                    fminf(fmaxf(SH_C0 * dc[1] + 0.5f, 0.0f), 1.0f),
                    fminf(fmaxf(SH_C0 * dc[2] + 0.5f, 0.0f), 1.0f),
                    opacity
                );
            } else {
                s_colorOpacity[threadIdxLinear] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
        // Evaluate Gaussians from shared memory
        if (pixelValid && accumulatedAlpha < 0.99f) {
            for (int i = 0; i < batchSize; ++i) {
                float2 meanNdc = s_means[i];
                float3 covInv = s_covInv[i];
                float4 colorOp = s_colorOpacity[i];
                
                // Skip invalid Gaussians
                if (colorOp.w <= 0.0f) continue;
                if (covInv.x == 0.0f && covInv.y == 0.0f && covInv.z == 0.0f) continue;
                
                float dx = ndcX - meanNdc.x;
                float dy = ndcY - meanNdc.y;
                
                float mahalDistSq = dx * dx * covInv.x + 2.0f * dx * dy * covInv.y + dy * dy * covInv.z;
                if (mahalDistSq > 9.0f) continue;
                
                float gaussianWeight = expf(-0.5f * mahalDistSq);
                float alpha = colorOp.w * gaussianWeight;
                if (alpha < 1.0f / 255.0f) continue;
                
                float oneMinusAccum = 1.0f - accumulatedAlpha;
                pixelColor.x += oneMinusAccum * colorOp.x * alpha;
                pixelColor.y += oneMinusAccum * colorOp.y * alpha;
                pixelColor.z += oneMinusAccum * colorOp.z * alpha;
                accumulatedAlpha += oneMinusAccum * alpha;
                
                if (accumulatedAlpha >= 0.99f) break;
            }
        }
        
        __syncthreads();
    }
    
    // Write final pixel color
    if (pixelValid) {
        pixelColor.x = fminf(fmaxf(pixelColor.x, 0.0f), 1.0f);
        pixelColor.y = fminf(fmaxf(pixelColor.y, 0.0f), 1.0f);
        pixelColor.z = fminf(fmaxf(pixelColor.z, 0.0f), 1.0f);
        pixelColor.w = 1.0f;
        
        int flippedX = width - 1 - pixelX;
        int flippedY = height - 1 - pixelY;
        framebuffer[flippedY * width + flippedX] = pixelColor;
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

CudaRenderer::CudaRenderer() 
    : d_means(nullptr)
    , d_scales(nullptr)
    , d_rotations(nullptr)
    , d_opacities(nullptr)
    , d_sh_dc(nullptr)
    , d_sh_rest(nullptr)
    , d_covariances(nullptr)
    , d_meansScreen(nullptr)
    , d_covariancesScreen(nullptr)
    , d_depths(nullptr)
    , d_visibleToOriginal(nullptr)
    , d_tileIndices(nullptr)
    , d_sortKeys(nullptr)
    , d_sortedIndices(nullptr)
    , d_tileCounts(nullptr)
    , d_tileOffsets(nullptr)
    , maxDuplicateEntries(0)
    , d_tileRanges(nullptr)
    , numGaussians(0)
    , numVisibleGaussians(0)
    , width(0)
    , height(0)
    , numTilesX(0)
    , numTilesY(0)
    , framebufferRegistered(false)
{
}

CudaRenderer::~CudaRenderer() {
    cleanup();
}

void CudaRenderer::init(int width, int height) {
    this->width = width;
    this->height = height;
    
    // Initialize CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        exit(1);
    }
    
    // Set device (use device 0)
    CUDA_CHECK(cudaSetDevice(0));
    
    // Initialize tile grid (16x16 pixel tiles)
    constexpr int TILE_SIZE = 16;
    numTilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
    numTilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
    
    // Copy tile/screen dimensions to constant memory for GPU kernels
    CUDA_CHECK(cudaMemcpyToSymbol(d_numTilesX, &numTilesX, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_numTilesY, &numTilesY, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
}

void CudaRenderer::uploadGaussians(const std::vector<GaussianSplat>& gaussians) {
    numGaussians = static_cast<int>(gaussians.size());
    
    if (numGaussians == 0) {
        return;
    }
    
    // Allocate GPU memory (Structure of Arrays)
    CUDA_CHECK(cudaMalloc(&d_means, numGaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, numGaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotations, numGaussians * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacities, numGaussians * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_dc, numGaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh_rest, numGaussians * 45 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariances, numGaussians * 9 * sizeof(float)));
    
    // Convert from AoS to SoA and upload
    std::vector<float> means(numGaussians * 3);
    std::vector<float> scales(numGaussians * 3);
    std::vector<float> rotations(numGaussians * 4);
    std::vector<float> opacities(numGaussians);
    std::vector<float> sh_dc(numGaussians * 3);
    std::vector<float> sh_rest(numGaussians * 45);
    std::vector<float> covariances(numGaussians * 9);
    
    for (int i = 0; i < numGaussians; ++i) {
        const auto& g = gaussians[i];
        
        // Means
        means[i * 3 + 0] = g.mean.x;
        means[i * 3 + 1] = g.mean.y;
        means[i * 3 + 2] = g.mean.z;
        
        // Scales
        scales[i * 3 + 0] = g.scale.x;
        scales[i * 3 + 1] = g.scale.y;
        scales[i * 3 + 2] = g.scale.z;
        
        // Rotations (quaternion)
        rotations[i * 4 + 0] = g.rotation.x;
        rotations[i * 4 + 1] = g.rotation.y;
        rotations[i * 4 + 2] = g.rotation.z;
        rotations[i * 4 + 3] = g.rotation.w;
        
        // Opacities
        opacities[i] = g.opacity;
        
        // SH DC
        sh_dc[i * 3 + 0] = g.sh_dc[0];
        sh_dc[i * 3 + 1] = g.sh_dc[1];
        sh_dc[i * 3 + 2] = g.sh_dc[2];
        
        // SH Rest
        for (int j = 0; j < 45; ++j) {
            sh_rest[i * 45 + j] = g.sh_rest[j];
        }
        
        // Covariances (3x3 matrix, row-major)
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                covariances[i * 9 + row * 3 + col] = g.covariance[row][col];
            }
        }
    }
    
    // Upload to GPU
    CUDA_CHECK(cudaMemcpy(d_means, means.data(), numGaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), numGaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotations, rotations.data(), numGaussians * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities, opacities.data(), numGaussians * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_dc, sh_dc.data(), numGaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sh_rest, sh_rest.data(), numGaussians * 45 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covariances, covariances.data(), numGaussians * 9 * sizeof(float), cudaMemcpyHostToDevice));
}

void CudaRenderer::registerFramebuffer(GLuint framebuffer) {
    if (framebufferRegistered) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cudaFramebufferResource));
    }
    
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaFramebufferResource, framebuffer, 
                                          GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    framebufferRegistered = true;
}

// GPU Culling Kernel - one thread per Gaussian
__global__ void cullKernel(
    const float* means, int numGaussians,
    const float* viewProjMatrix,  // 4x4 matrix as 16 floats (column-major)
    int* visibilityFlags,  // Output: 1 if visible, 0 if not
    float kFrustumMargin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numGaussians) return;
    
    // Load mean position
    float mx = means[idx * 3 + 0];
    float my = means[idx * 3 + 1];
    float mz = means[idx * 3 + 2];
    
    // Multiply by viewProj matrix (column-major order)
    float clipX = viewProjMatrix[0] * mx + viewProjMatrix[4] * my + viewProjMatrix[8] * mz + viewProjMatrix[12];
    float clipY = viewProjMatrix[1] * mx + viewProjMatrix[5] * my + viewProjMatrix[9] * mz + viewProjMatrix[13];
    float clipZ = viewProjMatrix[2] * mx + viewProjMatrix[6] * my + viewProjMatrix[10] * mz + viewProjMatrix[14];
    float clipW = viewProjMatrix[3] * mx + viewProjMatrix[7] * my + viewProjMatrix[11] * mz + viewProjMatrix[15];
    
    // Behind camera check
    if (clipW <= 0.0f) {
        visibilityFlags[idx] = 0;
        return;
    }
    
    // NDC conversion and frustum check
    float ndcX = clipX / clipW;
    float ndcY = clipY / clipW;
    float ndcZ = clipZ / clipW;
    
    if (fabsf(ndcX) > 1.0f + kFrustumMargin ||
        fabsf(ndcY) > 1.0f + kFrustumMargin ||
        ndcZ < -1.0f - kFrustumMargin ||
        ndcZ > 1.0f + kFrustumMargin) {
        visibilityFlags[idx] = 0;
        return;
    }
    
    visibilityFlags[idx] = 1;
}

// Algorithm 2, Step 1: CullGaussian(p, V)
// Frustum culling with GPU-parallel compaction
void CudaRenderer::cullGaussians(const glm::mat4& viewProj, std::vector<int>& visibleIndices) {
    visibleIndices.clear();
    
    constexpr float kFrustumMargin = 0.1f;
    
    // Allocate GPU buffers for culling if needed
    static float* d_viewProjMatrix = nullptr;
    static int* d_visibilityFlags = nullptr;
    static int* d_indices = nullptr;           // Input: [0, 1, 2, ..., N-1]
    static int* d_compactedIndices = nullptr;  // Output: compacted visible indices
    static int* d_numVisible = nullptr;        // Output count
    static int lastNumGaussians = 0;
    
    if (d_viewProjMatrix == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_viewProjMatrix, 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_numVisible, sizeof(int)));
    }
    if (d_visibilityFlags == nullptr || lastNumGaussians != numGaussians) {
        if (d_visibilityFlags) cudaFree(d_visibilityFlags);
        if (d_indices) cudaFree(d_indices);
        if (d_compactedIndices) cudaFree(d_compactedIndices);
        
        CUDA_CHECK(cudaMalloc(&d_visibilityFlags, numGaussians * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_indices, numGaussians * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_compactedIndices, numGaussians * sizeof(int)));
        
        // Initialize d_indices to [0, 1, 2, ..., N-1] using thrust::sequence
        thrust::device_ptr<int> d_indices_ptr(d_indices);
        thrust::sequence(d_indices_ptr, d_indices_ptr + numGaussians);
        
        lastNumGaussians = numGaussians;
    }
    
    // Upload viewProj matrix to GPU
    CUDA_CHECK(cudaMemcpy(d_viewProjMatrix, glm::value_ptr(viewProj), 16 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch culling kernel
    int blockSize = 256;
    int numBlocks = (numGaussians + blockSize - 1) / blockSize;
    cullKernel<<<numBlocks, blockSize>>>(d_means, numGaussians, d_viewProjMatrix, d_visibilityFlags, kFrustumMargin);
    CUDA_CHECK(cudaGetLastError());
    
    // GPU-parallel compaction using Thrust
    thrust::device_ptr<int> d_indices_ptr(d_indices);
    thrust::device_ptr<int> d_flags_ptr(d_visibilityFlags);
    thrust::device_ptr<int> d_output_ptr(d_compactedIndices);
    
    // Stream compaction: copy indices where flag == 1
    auto result_end = thrust::copy_if(
        d_indices_ptr,                    // Input begin
        d_indices_ptr + numGaussians,     // Input end
        d_flags_ptr,                      // Stencil (flags)
        d_output_ptr,                     // Output
        thrust::identity<int>()           // Predicate: flag != 0
    );
    
    // Get number of visible Gaussians
    int numVisible = result_end - d_output_ptr;
    numVisibleGaussians = numVisible;
    
    // Download compacted indices
    visibleIndices.resize(numVisible);
    if (numVisible > 0) {
        CUDA_CHECK(cudaMemcpy(visibleIndices.data(), d_compactedIndices, 
                              numVisible * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

// GPU Kernel for screen-space transform
__global__ void transformToScreenSpaceKernel(
    const float* means,           // World-space means [N x 3]
    const float* covariances,     // World-space covariances [N x 9]
    const int* visibleIndices,    // Indices of visible Gaussians
    int numVisible,
    const float* viewMatrix,      // 4x4 view matrix (column-major)
    const float* projMatrix,      // 4x4 projection matrix (column-major)
    float* meansScreen,           // Output: UV coords [N x 2]
    float* covariancesScreen,     // Output: 2D covariance [N x 4]
    float* depths,                // Output: camera-space depth
    int* visibleToOriginal        // Output: mapping back to original index
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVisible) return;
    
    int origIdx = visibleIndices[i];
    visibleToOriginal[i] = origIdx;
    
    // Load world-space mean
    float mx = means[origIdx * 3 + 0];
    float my = means[origIdx * 3 + 1];
    float mz = means[origIdx * 3 + 2];
    
    // Transform to camera space (view * mean)
    float camX = viewMatrix[0] * mx + viewMatrix[4] * my + viewMatrix[8] * mz + viewMatrix[12];
    float camY = viewMatrix[1] * mx + viewMatrix[5] * my + viewMatrix[9] * mz + viewMatrix[13];
    float camZ = viewMatrix[2] * mx + viewMatrix[6] * my + viewMatrix[10] * mz + viewMatrix[14];
    
    depths[i] = camZ;
    
    // Compute viewProj for projection
    float vp[16];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            vp[r + c*4] = 0;
            for (int k = 0; k < 4; k++) {
                vp[r + c*4] += projMatrix[r + k*4] * viewMatrix[k + c*4];
            }
        }
    }
    
    // Project to clip space
    float clipX = vp[0] * mx + vp[4] * my + vp[8] * mz + vp[12];
    float clipY = vp[1] * mx + vp[5] * my + vp[9] * mz + vp[13];
    float clipW = vp[3] * mx + vp[7] * my + vp[11] * mz + vp[15];
    
    // NDC to UV (standard conversion, flip handled in framebuffer copy)
    float ndcX = clipX / clipW;
    float ndcY = clipY / clipW;
    meansScreen[i * 2 + 0] = ndcX * 0.5f + 0.5f;
    meansScreen[i * 2 + 1] = ndcY * 0.5f + 0.5f;
    
    // ========== PROPER 2D COVARIANCE PROJECTION ==========
    // Step 1: Load full 3x3 world covariance (row-major)
    float cov3d[9];
    for (int j = 0; j < 9; j++) {
        cov3d[j] = covariances[origIdx * 9 + j];
    }
    
    // Step 2: Extract 3x3 rotation from view matrix (column-major storage)
    float R[9];
    R[0] = viewMatrix[0]; R[1] = viewMatrix[4]; R[2] = viewMatrix[8];
    R[3] = viewMatrix[1]; R[4] = viewMatrix[5]; R[5] = viewMatrix[9];
    R[6] = viewMatrix[2]; R[7] = viewMatrix[6]; R[8] = viewMatrix[10];
    
    // Step 3: Transform covariance to camera space: Σ_cam = R * Σ_world * R^T
    // First compute temp = R * Σ_world
    float temp[9];
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            temp[row * 3 + col] = 
                R[row * 3 + 0] * cov3d[0 * 3 + col] +
                R[row * 3 + 1] * cov3d[1 * 3 + col] +
                R[row * 3 + 2] * cov3d[2 * 3 + col];
        }
    }
    // Then compute Σ_cam = temp * R^T
    float covCam[9];
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            covCam[row * 3 + col] = 
                temp[row * 3 + 0] * R[col * 3 + 0] +
                temp[row * 3 + 1] * R[col * 3 + 1] +
                temp[row * 3 + 2] * R[col * 3 + 2];
        }
    }
    
    // Step 4: Compute Jacobian of perspective projection
    // For perspective projection: x' = fx * X/Z, y' = fy * Y/Z
    // Jacobian J = [fx/Z,    0,   -fx*X/Z^2]
    //              [0,    fy/Z,   -fy*Y/Z^2]
    // We need focal lengths from projection matrix: fx = P[0][0], fy = P[1][1]
    float fx = projMatrix[0];  // P[0][0]
    float fy = projMatrix[5];  // P[1][1]
    
    float invZ = 1.0f / (camZ + 0.0001f);
    float invZ2 = invZ * invZ;
    
    // J is 2x3:
    // J[0] = [fx*invZ, 0, -fx*camX*invZ2]
    // J[1] = [0, fy*invZ, -fy*camY*invZ2]
    float J00 = fx * invZ;
    float J02 = -fx * camX * invZ2;
    float J11 = fy * invZ;
    float J12 = -fy * camY * invZ2;
    
    // Step 5: Compute 2D covariance: Σ_2D = J * Σ_cam * J^T
    // First compute JΣ (2x3 * 3x3 = 2x3)
    float JS[6]; // 2x3
    // Row 0: [J00, 0, J02] * Σ_cam
    JS[0] = J00 * covCam[0] + J02 * covCam[6];
    JS[1] = J00 * covCam[1] + J02 * covCam[7];
    JS[2] = J00 * covCam[2] + J02 * covCam[8];
    // Row 1: [0, J11, J12] * Σ_cam
    JS[3] = J11 * covCam[3] + J12 * covCam[6];
    JS[4] = J11 * covCam[4] + J12 * covCam[7];
    JS[5] = J11 * covCam[5] + J12 * covCam[8];
    
    // Then compute (JΣ) * J^T (2x3 * 3x2 = 2x2)
    // J^T = [J00, 0  ]
    //       [0,   J11]
    //       [J02, J12]
    float cov2d_00 = JS[0] * J00 + JS[2] * J02;
    float cov2d_01 = JS[1] * J11 + JS[2] * J12;  // off-diagonal
    float cov2d_11 = JS[4] * J11 + JS[5] * J12;
    
    // No covariance flips needed - we flip the framebuffer output instead
    
    // Add small regularization to ensure positive definiteness
    constexpr float kMinCov = 1e-6f;
    cov2d_00 = fmaxf(cov2d_00, kMinCov);
    cov2d_11 = fmaxf(cov2d_11, kMinCov);
    
    // Store as [a, b, b, c] for symmetric matrix [[a,b],[b,c]]
    covariancesScreen[i * 4 + 0] = cov2d_00;
    covariancesScreen[i * 4 + 1] = cov2d_01;
    covariancesScreen[i * 4 + 2] = cov2d_01;  // symmetric
    covariancesScreen[i * 4 + 3] = cov2d_11;
}

// Count how many tiles each Gaussian overlaps
__global__ void countTilesPerGaussianKernel(
    const float* meansScreen,      // [numVisible * 2] - UV coordinates
    const float* covariancesScreen, // [numVisible * 4] - 2x2 covariance
    int numVisible,
    int* tileCounts                // Output: number of tiles per Gaussian
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVisible) return;
    
    constexpr int TILE_SIZE_LOCAL = 16;
    constexpr float SIGMA_MULT = 3.0f;
    
    float u = meansScreen[i * 2 + 0];
    float v = meansScreen[i * 2 + 1];
    
    // Skip Gaussians outside screen
    if (u < -0.5f || u > 1.5f || v < -0.5f || v > 1.5f) {
        tileCounts[i] = 0;
        return;
    }
    
    // Get covariance and compute screen-space radius (3-sigma)
    float cov_xx = covariancesScreen[i * 4 + 0];
    float cov_yy = covariancesScreen[i * 4 + 3];
    
    // Radius in NDC space, converted to pixels
    float radiusNDC = SIGMA_MULT * sqrtf(fmaxf(cov_xx, cov_yy));
    float radiusPixX = radiusNDC * 0.5f * d_width;
    float radiusPixY = radiusNDC * 0.5f * d_height;
    
    // Clamp radius
    radiusPixX = fmaxf(radiusPixX, 1.0f);
    radiusPixY = fmaxf(radiusPixY, 1.0f);
    radiusPixX = fminf(radiusPixX, (float)(TILE_SIZE_LOCAL * 8));
    radiusPixY = fminf(radiusPixY, (float)(TILE_SIZE_LOCAL * 8));
    
    // Convert center UV to pixel coordinates
    float centerPixX = u * d_width;
    float centerPixY = v * d_height;
    
    // Compute bounding box in tile coordinates
    int minTileX = (int)floorf((centerPixX - radiusPixX) / TILE_SIZE_LOCAL);
    int maxTileX = (int)floorf((centerPixX + radiusPixX) / TILE_SIZE_LOCAL);
    int minTileY = (int)floorf((centerPixY - radiusPixY) / TILE_SIZE_LOCAL);
    int maxTileY = (int)floorf((centerPixY + radiusPixY) / TILE_SIZE_LOCAL);
    
    // Clamp to valid tile range
    minTileX = max(minTileX, 0);
    maxTileX = min(maxTileX, d_numTilesX - 1);
    minTileY = max(minTileY, 0);
    maxTileY = min(maxTileY, d_numTilesY - 1);
    
    // Count tiles
    int numTiles = (maxTileX - minTileX + 1) * (maxTileY - minTileY + 1);
    tileCounts[i] = numTiles;
}

// Scatter entries to output arrays using prefix sum offsets
__global__ void scatterEntriesKernel(
    const float* meansScreen,
    const float* covariancesScreen,
    const float* depths,
    int numVisible,
    const int* tileOffsets,        // Prefix sum of tile counts (exclusive scan)
    unsigned int* sortKeys,        // Output: sort keys (tileId << 16 | depth)
    int* sortedIndices             // Output: Gaussian indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVisible) return;
    
    constexpr int TILE_SIZE_LOCAL = 16;
    constexpr float SIGMA_MULT = 3.0f;
    
    float u = meansScreen[i * 2 + 0];
    float v = meansScreen[i * 2 + 1];
    
    // Skip Gaussians outside screen (count was 0)
    if (u < -0.5f || u > 1.5f || v < -0.5f || v > 1.5f) {
        return;
    }
    
    float depth = depths[i];
    
    // Get covariance and compute screen-space radius
    float cov_xx = covariancesScreen[i * 4 + 0];
    float cov_yy = covariancesScreen[i * 4 + 3];
    
    float radiusNDC = SIGMA_MULT * sqrtf(fmaxf(cov_xx, cov_yy));
    float radiusPixX = radiusNDC * 0.5f * d_width;
    float radiusPixY = radiusNDC * 0.5f * d_height;
    
    radiusPixX = fmaxf(radiusPixX, 1.0f);
    radiusPixY = fmaxf(radiusPixY, 1.0f);
    radiusPixX = fminf(radiusPixX, (float)(TILE_SIZE_LOCAL * 8));
    radiusPixY = fminf(radiusPixY, (float)(TILE_SIZE_LOCAL * 8));
    
    float centerPixX = u * d_width;
    float centerPixY = v * d_height;
    
    int minTileX = (int)floorf((centerPixX - radiusPixX) / TILE_SIZE_LOCAL);
    int maxTileX = (int)floorf((centerPixX + radiusPixX) / TILE_SIZE_LOCAL);
    int minTileY = (int)floorf((centerPixY - radiusPixY) / TILE_SIZE_LOCAL);
    int maxTileY = (int)floorf((centerPixY + radiusPixY) / TILE_SIZE_LOCAL);
    
    minTileX = max(minTileX, 0);
    maxTileX = min(maxTileX, d_numTilesX - 1);
    minTileY = max(minTileY, 0);
    maxTileY = min(maxTileY, d_numTilesY - 1);
    
    // Prepare depth bits for sort key
    unsigned int depthBits = __float_as_uint(depth);
    depthBits ^= 0x80000000;  // Flip sign bit for proper float sorting
    
    // Get output position from prefix sum
    int outputIdx = tileOffsets[i];
    
    // Write entries for all tiles this Gaussian overlaps
    for (int ty = minTileY; ty <= maxTileY; ++ty) {
        for (int tx = minTileX; tx <= maxTileX; ++tx) {
            int tileIdx = ty * d_numTilesX + tx;
            
            // Create sort key: tile_id (upper 16 bits) | depth (lower 16 bits)
            unsigned int sortKey = (static_cast<unsigned int>(tileIdx) << 16) | (depthBits >> 16);
            
            sortKeys[outputIdx] = sortKey;
            sortedIndices[outputIdx] = i;  // Store Gaussian index
            outputIdx++;
        }
    }
}

// Initialize tile ranges to invalid (-1)
__global__ void initTileRangesKernel(int* tileRanges, int numTiles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTiles) return;
    
    tileRanges[i * 2 + 0] = -1;  // start
    tileRanges[i * 2 + 1] = -1;  // end
}

// Detect tile boundaries in sorted key array
__global__ void detectTileBoundariesKernel(
    const unsigned int* sortedKeys,
    int* tileRanges,
    int numEntries,
    int numTiles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numEntries) return;
    
    // Extract tile ID from sort key (upper 16 bits)
    int myTile = sortedKeys[i] >> 16;
    
    // Bounds check
    if (myTile < 0 || myTile >= numTiles) return;
    
    // Check if I'm the start of a tile (first entry OR different from previous)
    if (i == 0) {
        tileRanges[myTile * 2 + 0] = i;  // First entry is always a tile start
    } else {
        int prevTile = sortedKeys[i - 1] >> 16;
        if (myTile != prevTile) {
            tileRanges[myTile * 2 + 0] = i;  // I'm the start of my tile
        }
    }
    
    // Check if I'm the end of a tile (last entry OR different from next)
    if (i == numEntries - 1) {
        tileRanges[myTile * 2 + 1] = i + 1;  // Last entry ends its tile
    } else {
        int nextTile = sortedKeys[i + 1] >> 16;
        if (myTile != nextTile) {
            tileRanges[myTile * 2 + 1] = i + 1;  // I'm the end of my tile (exclusive)
        }
    }
}

// Algorithm 2, Step 2: ScreenspaceGaussians(M, S, V)
// Transform world-space Gaussians to screen space - GPU parallelized
void CudaRenderer::transformToScreenSpace(const glm::mat4& view, const glm::mat4& projection,
                                         const std::vector<int>& visibleIndices) {
    if (visibleIndices.empty()) return;
    
    size_t maxVisible = visibleIndices.size();
    
    // Allocate screen-space buffers on first call
    static float* d_viewMatrix = nullptr;
    static float* d_projMatrix = nullptr;
    static int* d_visibleIndicesGPU = nullptr;
    static int lastAllocSize = 0;
    
    if (d_meansScreen == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_meansScreen, numGaussians * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_covariancesScreen, numGaussians * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_depths, numGaussians * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_visibleToOriginal, numGaussians * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_viewMatrix, 16 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_projMatrix, 16 * sizeof(float)));
    }
    
    if (d_visibleIndicesGPU == nullptr || lastAllocSize < (int)maxVisible) {
        if (d_visibleIndicesGPU) cudaFree(d_visibleIndicesGPU);
        CUDA_CHECK(cudaMalloc(&d_visibleIndicesGPU, maxVisible * sizeof(int)));
        lastAllocSize = (int)maxVisible;
    }
    
    // Upload matrices and visible indices
    CUDA_CHECK(cudaMemcpy(d_viewMatrix, glm::value_ptr(view), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projMatrix, glm::value_ptr(projection), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visibleIndicesGPU, visibleIndices.data(), maxVisible * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = ((int)maxVisible + blockSize - 1) / blockSize;
    transformToScreenSpaceKernel<<<numBlocks, blockSize>>>(
        d_means, d_covariances, d_visibleIndicesGPU, (int)maxVisible,
        d_viewMatrix, d_projMatrix,
        d_meansScreen, d_covariancesScreen, d_depths, d_visibleToOriginal
    );
    CUDA_CHECK(cudaGetLastError());
}

// Algorithm 2, Step 3: CreateTiles(w, h)
void CudaRenderer::createTiles() {
    // Tiles are already created in init() - just verify
    // Tile grid: numTilesX x numTilesY tiles of size 16x16
}

// Algorithm 2, Step 4: DuplicateWithKeys(M', T)
// Assign splats to tiles and create depth keys for sorting
void CudaRenderer::duplicateWithKeys(const std::vector<int>& visibleIndices) {
    if (visibleIndices.empty()) return;
    
    int numVisible = static_cast<int>(visibleIndices.size());
    
    // Allocate tile count/offset buffers if needed
    static int allocatedCountSize = 0;
    if (d_tileCounts == nullptr || allocatedCountSize < numVisible) {
        if (d_tileCounts) cudaFree(d_tileCounts);
        if (d_tileOffsets) cudaFree(d_tileOffsets);
        
        int newSize = numVisible * 2;  // Extra space for growth
        CUDA_CHECK(cudaMalloc(&d_tileCounts, newSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tileOffsets, newSize * sizeof(int)));
        allocatedCountSize = newSize;
    }
    
    // Count tiles per Gaussian
    int blockSize = 256;
    int numBlocks = (numVisible + blockSize - 1) / blockSize;
    
    countTilesPerGaussianKernel<<<numBlocks, blockSize>>>(
        d_meansScreen,
        d_covariancesScreen,
        numVisible,
        d_tileCounts
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Prefix sum to get output positions
    thrust::device_ptr<int> d_counts_ptr(d_tileCounts);
    thrust::device_ptr<int> d_offsets_ptr(d_tileOffsets);
    
    // Exclusive scan: offsets[i] = sum(counts[0..i-1])
    thrust::exclusive_scan(d_counts_ptr, d_counts_ptr + numVisible, d_offsets_ptr);
    
    // Get total number of entries (last offset + last count)
    int lastOffset, lastCount;
    CUDA_CHECK(cudaMemcpy(&lastOffset, d_tileOffsets + numVisible - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastCount, d_tileCounts + numVisible - 1, sizeof(int), cudaMemcpyDeviceToHost));
    numVisibleGaussians = lastOffset + lastCount;
    
    if (numVisibleGaussians == 0) return;
    
    // Allocate output buffers if needed
    if (d_sortKeys == nullptr || maxDuplicateEntries < numVisibleGaussians) {
        if (d_sortKeys) cudaFree(d_sortKeys);
        if (d_sortedIndices) cudaFree(d_sortedIndices);
        if (d_tileIndices) cudaFree(d_tileIndices);
        
        int newSize = numVisibleGaussians * 2;  // Extra space
        CUDA_CHECK(cudaMalloc(&d_sortKeys, newSize * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_sortedIndices, newSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tileIndices, newSize * sizeof(int)));
        maxDuplicateEntries = newSize;
    }
    
    // Scatter entries to output arrays
    scatterEntriesKernel<<<numBlocks, blockSize>>>(
        d_meansScreen,
        d_covariancesScreen,
        d_depths,
        numVisible,
        d_tileOffsets,
        d_sortKeys,
        d_sortedIndices
    );
    CUDA_CHECK(cudaGetLastError());
}

// Algorithm 2, Step 5: SortByKeys(K, L)
// Global sort by keys (tile_id, depth) - GPU accelerated with Thrust
void CudaRenderer::sortByKeys() {
    if (numVisibleGaussians == 0) return;
    
    // Ensure buffers are allocated
    if (d_sortedIndices == nullptr || d_sortKeys == nullptr) {
        std::cerr << "Error: sort buffers not allocated!" << std::endl;
        return;
    }
    
    // Wrap raw pointers with thrust device pointers
    thrust::device_ptr<unsigned int> d_keys_ptr(d_sortKeys);
    thrust::device_ptr<int> d_indices_ptr(d_sortedIndices);
    
    // Sort: d_sortedIndices already contains the visible Gaussian indices
    // (set by duplicateWithKeys), we just need to sort them by the keys
    // This rearranges both d_sortKeys and d_sortedIndices together
    thrust::sort_by_key(d_keys_ptr, d_keys_ptr + numVisibleGaussians, d_indices_ptr);
}

// Algorithm 2, Step 6: IdentifyTileRanges(T, K)
// Find start and end indices for each tile in sorted list
void CudaRenderer::identifyTileRanges() {
    if (numVisibleGaussians == 0) return;
    
    int numTiles = numTilesX * numTilesY;
    if (d_tileRanges == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_tileRanges, numTiles * 2 * sizeof(int)));
    }
    
    int blockSize = 256;
    
    // Initialize all tile ranges to -1
    int initBlocks = (numTiles + blockSize - 1) / blockSize;
    initTileRangesKernel<<<initBlocks, blockSize>>>(d_tileRanges, numTiles);
    CUDA_CHECK(cudaGetLastError());
    
    // Detect tile boundaries
    int detectBlocks = (numVisibleGaussians + blockSize - 1) / blockSize;
    detectTileBoundariesKernel<<<detectBlocks, blockSize>>>(
        d_sortKeys,
        d_tileRanges,
        numVisibleGaussians,
        numTiles
    );
    CUDA_CHECK(cudaGetLastError());
    
    // All data stays on GPU
}

// Algorithm 2, Step 7: BlendInOrder (per-tile blending)
// Launch CUDA kernel to blend all tiles
void CudaRenderer::blendTiles(const CameraState& camera) {
    if (numVisibleGaussians == 0 || !framebufferRegistered) return;
    
    // Map framebuffer for CUDA access
    cudaGraphicsMapResources(1, &cudaFramebufferResource, nullptr);
    cudaArray_t framebufferArray;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&framebufferArray, cudaFramebufferResource, 0, 0));
    
    // Allocate temporary framebuffer buffer
    float4* d_framebuffer;
    CUDA_CHECK(cudaMalloc(&d_framebuffer, width * height * sizeof(float4)));
    
    // Clear framebuffer to background color
    float4 bgColor = make_float4(0.07f, 0.07f, 0.09f, 1.0f);
    // Use a simple kernel to clear (or memset pattern)
    // For now, we'll let the blending kernel handle background
    
    // Launch blending kernel
    // Grid: one block per tile (numTilesX x numTilesY)
    // Block: one thread per pixel in tile (TILE_SIZE x TILE_SIZE)
    dim3 gridSize(numTilesX, numTilesY);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    
    // Get camera position for view direction
    float3 cameraPos = make_float3(camera.pos.x, camera.pos.y, camera.pos.z);
    
    // Launch blending kernel
    blendTilesKernel<<<gridSize, blockSize>>>(
        d_framebuffer, width, height,
        reinterpret_cast<const float2*>(d_meansScreen),
        d_covariancesScreen,
        d_opacities,
        d_sh_dc,
        d_sh_rest,
        d_means,
        d_visibleToOriginal,
        cameraPos,
        d_sortedIndices,
        d_tileRanges,
        numTilesX,
        numVisibleGaussians
    );
    
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy from buffer to OpenGL texture array
    cudaMemcpy2DToArray(
        framebufferArray, 0, 0,
        d_framebuffer, width * sizeof(float4),
        width * sizeof(float4), height,
        cudaMemcpyDeviceToDevice
    );
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_framebuffer));
    cudaGraphicsUnmapResources(1, &cudaFramebufferResource, nullptr);
}

#define BUILD_VERSION "BUILD_OPT4_001"
static const char* BUILD_TIMESTAMP = __DATE__ " " __TIME__;

// Algorithm 2: Main render function
static int frameCount = 0;

void CudaRenderer::render(const glm::mat4& view, const glm::mat4& projection, const CameraState& camera) {
    if (numGaussians == 0) return;
    
    // Initialize benchmark tracker on first frame
    static bool initialized = false;
    if (!initialized) {
        g_benchmark.config.warmupFrames = 30;
        g_benchmark.config.measureFrames = 100;
        g_benchmark.config.verbose = true;
        g_benchmark.init();
        initialized = true;
    }
    
    // Start benchmark frame
    g_benchmark.startFrame();
    
    mat4 viewProj = projection * view;
    
    // Step 1: CullGaussian(p, V)
    g_benchmark.recordCullStart();
    std::vector<int> visibleIndices;
    cullGaussians(viewProj, visibleIndices);
    cudaDeviceSynchronize();
    g_benchmark.recordCullEnd();
    
    // Early exit if nothing visible
    if (visibleIndices.empty()) {
        return;
    }
    
    // Step 2: ScreenspaceGaussians(M, S, V)
    g_benchmark.recordTransformStart();
    transformToScreenSpace(view, projection, visibleIndices);
    cudaDeviceSynchronize();
    g_benchmark.recordTransformEnd();
    
    // Step 3: CreateTiles(w, h)
    createTiles();
    
    // Step 4: DuplicateWithKeys(M', T)
    g_benchmark.recordDuplicateStart();
    duplicateWithKeys(visibleIndices);
    cudaDeviceSynchronize();
    g_benchmark.recordDuplicateEnd();
    
    // Step 5: SortByKeys(K, L)
    g_benchmark.recordSortStart();
    sortByKeys();
    cudaDeviceSynchronize();
    g_benchmark.recordSortEnd();
    
    // Step 6: IdentifyTileRanges(T, K)
    g_benchmark.recordRangeStart();
    identifyTileRanges();
    cudaDeviceSynchronize();
    g_benchmark.recordRangeEnd();
    
    // Step 7: BlendInOrder (per-tile blending)
    g_benchmark.recordBlendStart();
    blendTiles(camera);
    cudaDeviceSynchronize();
    g_benchmark.recordBlendEnd();
    
    // End benchmark frame
    g_benchmark.endFrame(numGaussians, static_cast<int>(visibleIndices.size()), numVisibleGaussians);
}

void CudaRenderer::cleanup() {
    if (framebufferRegistered) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cudaFramebufferResource));
        framebufferRegistered = false;
    }
    
    if (d_means) CUDA_CHECK(cudaFree(d_means));
    if (d_scales) CUDA_CHECK(cudaFree(d_scales));
    if (d_rotations) CUDA_CHECK(cudaFree(d_rotations));
    if (d_opacities) CUDA_CHECK(cudaFree(d_opacities));
    if (d_sh_dc) CUDA_CHECK(cudaFree(d_sh_dc));
    if (d_sh_rest) CUDA_CHECK(cudaFree(d_sh_rest));
    if (d_covariances) CUDA_CHECK(cudaFree(d_covariances));
    if (d_meansScreen) CUDA_CHECK(cudaFree(d_meansScreen));
    if (d_covariancesScreen) CUDA_CHECK(cudaFree(d_covariancesScreen));
    if (d_depths) CUDA_CHECK(cudaFree(d_depths));
    if (d_visibleToOriginal) CUDA_CHECK(cudaFree(d_visibleToOriginal));
    if (d_tileIndices) CUDA_CHECK(cudaFree(d_tileIndices));
    if (d_sortKeys) CUDA_CHECK(cudaFree(d_sortKeys));
    if (d_sortedIndices) CUDA_CHECK(cudaFree(d_sortedIndices));
    if (d_tileCounts) CUDA_CHECK(cudaFree(d_tileCounts));
    if (d_tileOffsets) CUDA_CHECK(cudaFree(d_tileOffsets));
    if (d_tileRanges) CUDA_CHECK(cudaFree(d_tileRanges));
    
    d_means = nullptr;
    d_scales = nullptr;
    d_rotations = nullptr;
    d_opacities = nullptr;
    d_sh_dc = nullptr;
    d_sh_rest = nullptr;
    d_covariances = nullptr;
    d_meansScreen = nullptr;
    d_covariancesScreen = nullptr;
    d_depths = nullptr;
    d_visibleToOriginal = nullptr;
    d_tileIndices = nullptr;
    d_sortKeys = nullptr;
    d_sortedIndices = nullptr;
    d_tileCounts = nullptr;
    d_tileOffsets = nullptr;
    d_tileRanges = nullptr;
}

