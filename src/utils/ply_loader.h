#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <limits>

// Load Gaussian splats from a PLY file
// Returns a vector of GaussianSplat structures
std::vector<GaussianSplat> loadGaussiansFromPly(
    const std::string& path,
    size_t maxCount = std::numeric_limits<size_t>::max()
);

