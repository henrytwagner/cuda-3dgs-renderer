#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>

using namespace glm;

// Camera state structure
struct CameraState {
    vec3 pos{0.0f, 0.0f, 1.5f};
    float yaw = -90.0f;
    float pitch = 0.0f;
    float fovDeg = 60.0f;
    float moveSpeed = 1.0f;
    float mouseSensitivity = 0.1f;
};

// Gaussian splat data structure (Array of Structures format)
struct GaussianSplat {
    vec3 mean;
    vec3 scale;
    quat rotation;
    float opacity;
    std::array<float, 3> sh_dc;
    std::array<float, 45> sh_rest;
    mat3 covariance;  // Precomputed from scale and rotation
};

// Legacy structure (kept for compatibility)
struct Gaussian3d {
    vec3 mean;
    mat3 cov;
    vec3 color{0.2f, 0.6f, 0.9f};
    float opacity{1.0f};
    std::array<float, 3> sh_dc{};
    std::array<float, 45> sh_rest{};
};

