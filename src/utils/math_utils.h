#pragma once

#include "types.h"
#include <glm/gtc/matrix_transform.hpp>

// Project 3D world mean to 2D UV coordinates
vec2 projectMeanToUV(const mat4& viewProj, const vec3& meanWorld);

// Transform 3D world-space covariance to camera space
mat3 worldToCameraCovariance(const mat3& viewRotation, const mat3& covWorld);

// Compute 2D screen-space covariance from 3D camera-space covariance
// Uses the full Jacobian of the projection transformation
mat2 cameraToScreenCovariance(const mat4& projection, const vec3& meanCam, const mat3& covCam);

// Create a simple axis-aligned 3D Gaussian for testing
Gaussian3d makeAxisAlignedGaussian3d(const vec3& mean, const vec3& stdDev);

