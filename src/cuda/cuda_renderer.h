#pragma once

#include <glad/glad.h>  // Must be included before cuda_gl_interop.h for GL types
#include "../utils/types.h"
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

// CUDA renderer for Gaussian splatting
// Implements tile-based rendering with binning, sorting, and blending

struct CudaRenderer {
    // GPU memory buffers (Structure of Arrays format)
    // These will be allocated on the GPU
    float* d_means;           // 3 floats per splat (world space)
    float* d_scales;          // 3 floats per splat
    float* d_rotations;       // 4 floats per splat (quaternion)
    float* d_opacities;       // 1 float per splat
    float* d_sh_dc;           // 3 floats per splat
    float* d_sh_rest;         // 45 floats per splat
    float* d_covariances;     // 9 floats per splat (3x3 matrix, row-major, world space)
    
    // Algorithm 2: Screen-space Gaussians (M', S')
    float* d_meansScreen;     // 2 floats per splat (UV coordinates)
    float* d_covariancesScreen; // 4 floats per splat (2x2 matrix, row-major)
    float* d_depths;          // 1 float per splat (camera-space Z for sorting)
    
    // Algorithm 2: Tile assignment and sorting
    int* d_visibleToOriginal; // Maps visible index → original Gaussian index
    int* d_tileIndices;       // Which tile each splat belongs to
    unsigned int* d_sortKeys; // Depth keys for sorting (32-bit: tile_id | depth)
    int* d_sortedIndices;     // Sorted indices (L in algorithm)
    
    // GPU-parallel duplicate buffers (Optimization 1)
    int* d_tileCounts;        // Number of tiles each Gaussian overlaps
    int* d_tileOffsets;       // Exclusive prefix sum of tileCounts
    int maxDuplicateEntries;  // Maximum allocated size for duplicate buffers
    
    // Algorithm 2: Tile ranges (R in algorithm)
    int* d_tileRanges;        // [start, end] pairs per tile (2 ints per tile)
    
    // Rendering state
    int numGaussians;
    int numVisibleGaussians;  // After culling
    int width, height;
    int numTilesX, numTilesY; // Tile grid dimensions
    
    // CUDA-OpenGL interop
    cudaGraphicsResource_t cudaFramebufferResource;
    bool framebufferRegistered;
    
    CudaRenderer();
    ~CudaRenderer();
    
    // Initialize renderer with screen dimensions
    void init(int width, int height);
    
    // Upload Gaussian data to GPU (converts from AoS to SoA)
    void uploadGaussians(const std::vector<GaussianSplat>& gaussians);
    
    // Register OpenGL framebuffer for CUDA access
    void registerFramebuffer(GLuint framebuffer);
    
    // Render frame (main entry point) - implements Algorithm 2
    void render(const glm::mat4& view, const glm::mat4& projection, const CameraState& camera);
    
    // Algorithm 2 helper functions
    void cullGaussians(const glm::mat4& viewProj, std::vector<int>& visibleIndices);
    void transformToScreenSpace(const glm::mat4& view, const glm::mat4& projection, 
                                 const std::vector<int>& visibleIndices);
    void createTiles();
    void duplicateWithKeys(const std::vector<int>& visibleIndices);
    void sortByKeys();
    void identifyTileRanges();
    void blendTiles(const CameraState& camera);
    
    // Cleanup
    void cleanup();
};
