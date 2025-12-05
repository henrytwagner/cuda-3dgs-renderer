#include "math_utils.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

using namespace glm;

vec2 projectMeanToUV(const mat4& viewProj, const vec3& meanWorld) {
    vec4 clip = viewProj * vec4(meanWorld, 1.0f);
    vec3 ndc = vec3(clip) / clip.w;
    return 0.5f * (vec2(ndc.x, ndc.y) + vec2(1.0f));
}

mat3 worldToCameraCovariance(const mat3& viewRotation, const mat3& worldCov) {
    return viewRotation * worldCov * transpose(viewRotation);
}

mat2 cameraToScreenCovariance(const mat4& projection, const vec3& meanCam, const mat3& covCam) {
    float z = meanCam.z;
    constexpr float kEpsilon = 1e-4f;
    if (abs(z) < kEpsilon) {
        z = (z >= 0.0f) ? kEpsilon : -kEpsilon;
    }

     // Pull needed projection coefficients (column-major layout)
     const float px = projection[0][0];
     const float py = projection[1][1];
     const float p02 = projection[2][0];
     const float p12 = projection[2][1];
     const float p22 = projection[2][2];
     const float p32 = projection[3][2];
     const float p23 = projection[2][3];
     const float p33 = projection[3][3];
 
     // Camera-space point
     const float x = meanCam.x;
     const float y = meanCam.y;
 
     // Clip-space w and its reciprocal
     const float w = p23 * z + p33;
     const float invW = 1.0f / w;
     const float invW2 = invW * invW;
 
     // Full 2x3 Jacobian J = ∂(uv)/∂(X,Y,Z)
     mat2x3 J;
     J[0][0] = 0.5f * px * invW;
     J[0][1] = 0.0f;
     J[0][2] = 0.5f * ((p02 * w - px * x) * invW2 + px * p32 * invW2 * z);
 
     J[1][0] = 0.0f;
     J[1][1] = 0.5f * py * invW;
     J[1][2] = 0.5f * ((p12 * w - py * y) * invW2 + py * p32 * invW2 * z);
 
     // Σ_screen = J * Σ_cam * J^T
     const vec3 c0 = covCam * J[0];
     const vec3 c1 = covCam * J[1];
 
    float maxRadius = 0.3f;
     mat2 covScreen;
     covScreen[0][0] = dot(J[0], c0);
     covScreen[0][1] = dot(J[0], c1);
     covScreen[1][0] = covScreen[0][1];
     covScreen[1][1] = dot(J[1], c1);
 
     return covScreen;
}

Gaussian3d makeAxisAlignedGaussian3d(const vec3& mean, const vec3& stdDevs) {
    Gaussian3d g;
    g.mean = mean;
    vec3 variances = stdDevs * stdDevs;
    g.cov = mat3(
        variances.x, 0.0f, 0.0f,
        0.0f, variances.y, 0.0f,
        0.0f, 0.0f, variances.z
    );
    return g;
}

