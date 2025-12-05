#include "ply_loader.h"
#include "tinyply.h"
#include <fstream>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

// Sigmoid function for opacity conversion (3DGS stores opacity in logit space)
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<GaussianSplat> loadGaussiansFromPly(const std::string& path,
                                                size_t maxCount)
{
    using namespace tinyply;

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open PLY file: " + path);
    }

    PlyFile file;
    file.parse_header(stream);
    
    // Print PLY file structure (useful for debugging incompatible files)
    std::cout << "PLY file elements:" << std::endl;
    for (const auto& elem : file.get_elements()) {
        std::cout << "  Element: " << elem.name << " (" << elem.size << " entries)" << std::endl;
        for (const auto& prop : elem.properties) {
            std::cout << "    Property: " << prop.name << std::endl;
        }
    }

    auto positions = file.request_properties_from_element("vertex", {"x", "y", "z"});
    
    // Some PLY files have normals (nx, ny, nz) - request them so tinyply can properly
    // skip them when parsing binary data, even though we don't use them
    std::shared_ptr<PlyData> normals;
    try {
        normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
        std::cout << "PLY file contains normals (will be ignored)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "No normals in PLY file (or error: " << e.what() << ")" << std::endl;
    }
    
    auto sh_dc     = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});

    std::vector<std::string> sh_rest_keys;
    sh_rest_keys.reserve(45);
    for (int i = 0; i < 45; ++i) {
        sh_rest_keys.push_back("f_rest_" + std::to_string(i));
    }
    auto sh_rest   = file.request_properties_from_element("vertex", sh_rest_keys);

    auto opacities = file.request_properties_from_element("vertex", {"opacity"});
    auto scales    = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
    auto rotations = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

    file.read(stream);

    const size_t count = std::min(positions->count, maxCount);
    std::vector<GaussianSplat> gaussians(count);

    const float* pos_ptr  = reinterpret_cast<float*>(positions->buffer.get());
    const float* dc_ptr   = reinterpret_cast<float*>(sh_dc->buffer.get());
    const float* rest_ptr = reinterpret_cast<float*>(sh_rest->buffer.get());
    const float* opa_ptr  = reinterpret_cast<float*>(opacities->buffer.get());
    const float* scl_ptr  = reinterpret_cast<float*>(scales->buffer.get());
    const float* rot_ptr  = reinterpret_cast<float*>(rotations->buffer.get());

    for (size_t i = 0; i < count; ++i) {
        GaussianSplat& g = gaussians[i];
        g.mean = vec3(pos_ptr[3*i + 0], pos_ptr[3*i + 1], pos_ptr[3*i + 2]);

        for (int c = 0; c < 3; ++c) g.sh_dc[c] = dc_ptr[3*i + c];
        std::copy(rest_ptr + 45*i, rest_ptr + 45*(i+1), g.sh_rest.begin());

        // Opacity is stored in logit-space in PLY, convert with sigmoid
        g.opacity = sigmoid(opa_ptr[i]);

        // Scales are stored in log-space in PLY, convert with exp
        vec3 logScale = vec3(scl_ptr[3*i + 0], scl_ptr[3*i + 1], scl_ptr[3*i + 2]);
        vec3 scale = vec3(std::exp(logScale.x), std::exp(logScale.y), std::exp(logScale.z));
        scale = glm::max(scale, vec3(1e-6f));  // Clamp to avoid zero
        g.scale = scale;

        // PLY stores rot_0=w, rot_1=x, rot_2=y, rot_3=z
        // GLM quat constructor is quat(w, x, y, z)
        g.rotation = quat(rot_ptr[4*i + 0], rot_ptr[4*i + 1], rot_ptr[4*i + 2], rot_ptr[4*i + 3]);
        g.rotation = normalize(g.rotation);

        // Compute covariance matrix from scale and rotation
        mat3 R = mat3_cast(g.rotation);
        mat3 S = mat3(scale.x * scale.x, 0.0f, 0.0f,
                      0.0f, scale.y * scale.y, 0.0f,
                      0.0f, 0.0f, scale.z * scale.z);
        g.covariance = R * S * transpose(R);
    }

    // Print scale and opacity statistics
    if (!gaussians.empty()) {
        vec3 minScale(FLT_MAX), maxScale(-FLT_MAX), avgScale(0);
        float minOpa = FLT_MAX, maxOpa = -FLT_MAX;
        for (const auto& g : gaussians) {
            minScale = glm::min(minScale, g.scale);
            maxScale = glm::max(maxScale, g.scale);
            avgScale += g.scale;
            minOpa = std::min(minOpa, g.opacity);
            maxOpa = std::max(maxOpa, g.opacity);
        }
        avgScale /= float(gaussians.size());
        std::cout << "Scale stats - min: (" << minScale.x << "," << minScale.y << "," << minScale.z 
                  << ") max: (" << maxScale.x << "," << maxScale.y << "," << maxScale.z 
                  << ") avg: (" << avgScale.x << "," << avgScale.y << "," << avgScale.z << ")" << std::endl;
        std::cout << "Opacity stats - min: " << minOpa << " max: " << maxOpa << std::endl;
    }

    return gaussians;
}

