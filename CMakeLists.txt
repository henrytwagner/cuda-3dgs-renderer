cmake_minimum_required(VERSION 3.20)

# Check for CUDA compiler
find_program(CUDA_NVCC_EXECUTABLE nvcc PATHS
    ENV CUDA_PATH
    ENV CUDA_HOME
    /usr/local/cuda
    /opt/cuda
    /usr/local/cuda-*
    /opt/cuda-*
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)

if(CUDA_NVCC_EXECUTABLE)
    enable_language(CUDA)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA found: ${CUDA_NVCC_EXECUTABLE}")
else()
    set(ENABLE_CUDA OFF)
    message(STATUS "CUDA not found - CUDA renderer will not be built")
    message(STATUS "Note: CUDA requires a CUDA-capable GPU and Windows/Linux")
endif()

project(cuda-3dgs-renderer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # Set CUDA architecture for RTX 4090 (Ada Lovelace = SM 89)
    # Also include SM 86 for RTX 30 series compatibility
    set(CMAKE_CUDA_ARCHITECTURES "86;89")
endif()

add_subdirectory(external/glfw)

add_subdirectory(external/glm)

add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# Utilities library
add_library(utils STATIC
    src/utils/ply_loader.cpp
    src/utils/math_utils.cpp
)
target_include_directories(utils PUBLIC 
    src/utils
    external/glm
)
target_sources(utils PRIVATE external/tinyply/tinyply.cpp)
target_include_directories(utils PRIVATE external/tinyply)


# CUDA renderer (only build if CUDA is available)
if(ENABLE_CUDA)
    add_executable(renderer_cuda 
        src/cuda/main_cuda.cpp
        src/cuda/cuda_renderer.cu
    )
    target_include_directories(renderer_cuda PRIVATE 
        external/glad/include
        external
        src/utils
        src/cuda
    )
    target_link_libraries(renderer_cuda PRIVATE 
        utils 
        glad 
        glfw 
        glm
    )
    
    # CUDA libraries are linked automatically when CUDA language is enabled
    if(APPLE)
        target_link_libraries(renderer_cuda PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreVideo" "-framework OpenGL")
    endif()
endif()
