#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cfloat>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Utility headers
#include "../utils/types.h"
#include "../utils/ply_loader.h"
#include "../utils/math_utils.h"

// CUDA renderer
#include "cuda_renderer.h"
#include "benchmark.h"

using namespace std;
using namespace glm;

// Window dimensions
constexpr int kWindowWidth = 720;
constexpr int kWindowHeight = 720;

// Global camera state
CameraState gCamera;
static bool gMouseCaptured = false;
static double gLastMouseX = 0.0, gLastMouseY = 0.0;

// Auto-benchmark mode globals
static bool gAutoBenchmark = false;
static vec3 gSceneCenter(0.0f);
static float gSceneRadius = 2.0f;
static int gBenchmarkFrame = 0;

// Camera parameters
const float kNearPlane = 0.1f;
const float kFarPlane = 10.0f;

void glfw_error_callback(int error, const char* description) {
    cerr << "GLFW Error " << error << ": " << description << '\n';
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gMouseCaptured) return;
    
    static bool firstMouse = true;
    if (firstMouse) {
        gLastMouseX = xpos;
        gLastMouseY = ypos;
        firstMouse = false;
    }
    
    float xoffset = static_cast<float>(xpos - gLastMouseX);
    float yoffset = static_cast<float>(gLastMouseY - ypos);
    gLastMouseX = xpos;
    gLastMouseY = ypos;
    
    xoffset *= gCamera.mouseSensitivity;
    yoffset *= gCamera.mouseSensitivity;
    
    gCamera.yaw -= xoffset;
    gCamera.pitch -= yoffset;
    
    if (gCamera.pitch > 89.0f) gCamera.pitch = 89.0f;
    if (gCamera.pitch < -89.0f) gCamera.pitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    gCamera.fovDeg -= static_cast<float>(yoffset);
    if (gCamera.fovDeg < 1.0f) gCamera.fovDeg = 1.0f;
    if (gCamera.fovDeg > 120.0f) gCamera.fovDeg = 120.0f;
}

void updateCameraPosition(GLFWwindow* window, float dt) {
    if (!gMouseCaptured) return;
    
    vec3 forward{
        cos(radians(gCamera.yaw)) * cos(radians(gCamera.pitch)),
        sin(radians(gCamera.pitch)),
        sin(radians(gCamera.yaw)) * cos(radians(gCamera.pitch))
    };
    forward = normalize(forward);
    
    vec3 right = normalize(cross(forward, vec3(0.0f, 1.0f, 0.0f)));
    vec3 up = cross(right, forward);
    
    vec3 moveDir(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) moveDir += forward;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) moveDir -= forward;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) moveDir += right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) moveDir -= right;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) moveDir += vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) moveDir -= vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) moveDir -= vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) moveDir += vec3(0.0f, 1.0f, 0.0f);
    
    if (length(moveDir) > 0.0f) {
        gCamera.pos += normalize(moveDir) * gCamera.moveSpeed * dt;
    }
}

// Scripted camera movement for automated benchmarking
void updateAutoBenchmarkCamera(int frame) {
    float t = frame / 130.0f;
    float orbitAngle = t * 360.0f * 1.5f;
    
    float distance;
    if (frame < 40) {
        distance = gSceneRadius * 1.8f;
    } else if (frame < 80) {
        float zoomT = (frame - 40) / 40.0f;
        distance = glm::mix(gSceneRadius * 1.8f, gSceneRadius * 1.0f, zoomT);
    } else {
        distance = gSceneRadius * 1.0f;
    }
    
    float pitch = sin(t * 3.14159f * 2.0f) * 15.0f;
    
    float x = cos(radians(orbitAngle)) * cos(radians(pitch)) * distance;
    float y = sin(radians(pitch)) * distance * 0.3f;
    float z = sin(radians(orbitAngle)) * cos(radians(pitch)) * distance;
    
    gCamera.pos = gSceneCenter + vec3(x, y, z);
    
    vec3 toCenter = normalize(gSceneCenter - gCamera.pos);
    gCamera.yaw = degrees(atan2(toCenter.z, toCenter.x));
    gCamera.pitch = degrees(asin(toCenter.y));
}

std::string getBasename(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
    size_t lastDot = filename.find_last_of(".");
    return (lastDot != std::string::npos) ? filename.substr(0, lastDot) : filename;
}

int main(int argc, char* argv[]) {
    std::string plyPath = "data/splats/cactus_splat3_30kSteps_464k_splats.ply";
    std::string outputDir = "benchmark_data";
    
    // Camera override flags
    bool hasCustomPos = false;
    vec3 customPos;
    bool hasCustomYaw = false;
    float customYaw = 0.0f;
    bool hasCustomPitch = false;
    float customPitch = 0.0f;
    bool hasCustomFov = false;
    float customFov = 60.0f;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0 || strcmp(argv[i], "-b") == 0) {
            gAutoBenchmark = true;
            cout << "Auto-benchmark mode enabled" << endl;
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                outputDir = argv[++i];
            }
        } else if (strcmp(argv[i], "--pos") == 0 || strcmp(argv[i], "--position") == 0) {
            if (i + 1 < argc) {
                // Parse "x,y,z" format
                std::string posStr = argv[++i];
                size_t comma1 = posStr.find(',');
                size_t comma2 = posStr.find(',', comma1 + 1);
                if (comma1 != std::string::npos && comma2 != std::string::npos) {
                    customPos.x = std::stof(posStr.substr(0, comma1));
                    customPos.y = std::stof(posStr.substr(comma1 + 1, comma2 - comma1 - 1));
                    customPos.z = std::stof(posStr.substr(comma2 + 1));
                    hasCustomPos = true;
                    cout << "Custom camera position: (" << customPos.x << ", " << customPos.y << ", " << customPos.z << ")" << endl;
                } else {
                    cerr << "Error: --pos requires format 'x,y,z'" << endl;
                    return EXIT_FAILURE;
                }
            }
        } else if (strcmp(argv[i], "--yaw") == 0) {
            if (i + 1 < argc) {
                customYaw = std::stof(argv[++i]);
                hasCustomYaw = true;
                cout << "Custom yaw: " << customYaw << "°" << endl;
            }
        } else if (strcmp(argv[i], "--pitch") == 0) {
            if (i + 1 < argc) {
                customPitch = std::stof(argv[++i]);
                hasCustomPitch = true;
                cout << "Custom pitch: " << customPitch << "°" << endl;
            }
        } else if (strcmp(argv[i], "--fov") == 0 || strcmp(argv[i], "--zoom") == 0) {
            if (i + 1 < argc) {
                customFov = std::stof(argv[++i]);
                hasCustomFov = true;
                cout << "Custom FOV: " << customFov << "°" << endl;
            }
        } else if (argv[i][0] != '-') {
            plyPath = argv[i];
        }
    }
    
    if (gAutoBenchmark) {
        g_benchmark.config.outputFile = outputDir + "/" + getBasename(plyPath) + "_benchmark.csv";
        g_benchmark.config.warmupFrames = 30;
        g_benchmark.config.measureFrames = 100;
    }
    
    glfwSetErrorCallback(glfw_error_callback);
    
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << '\n';
        return EXIT_FAILURE;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "CUDA Gaussian Renderer", nullptr, nullptr);
    if (!window) {
        cerr << "Failed to create GLFW window" << '\n';
        glfwTerminate();
        return EXIT_FAILURE;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cerr << "Failed to initialize GLAD" << '\n';
        return EXIT_FAILURE;
    }
    
    CudaRenderer cudaRenderer;
    cudaRenderer.init(kWindowWidth, kWindowHeight);
    
    std::vector<GaussianSplat> gaussians;
    try {
        gaussians = loadGaussiansFromPly(plyPath);
        if (!gaussians.empty()) {
            vec3 minPos(FLT_MAX), maxPos(-FLT_MAX);
            for (const auto& g : gaussians) {
                minPos = glm::min(minPos, g.mean);
                maxPos = glm::max(maxPos, g.mean);
            }
            vec3 center = (minPos + maxPos) * 0.5f;
            
            gSceneCenter = center;
            gSceneRadius = length(maxPos - minPos) * 0.5f;
            if (gSceneRadius < 0.5f) gSceneRadius = 2.0f;
            
            // Set default camera position
            if (!hasCustomPos) {
                gCamera.pos = center + vec3(0, 0, gSceneRadius * 1.5f);
            } else {
                gCamera.pos = customPos;
            }
            
            if (!hasCustomYaw) {
                gCamera.yaw = -90.0f;
            } else {
                gCamera.yaw = customYaw;
            }
            
            if (!hasCustomPitch) {
                gCamera.pitch = 0.0f;
            } else {
                gCamera.pitch = customPitch;
            }
            
            if (!hasCustomFov) {
                gCamera.fovDeg = 60.0f;
            } else {
                gCamera.fovDeg = customFov;
            }
            
            cudaRenderer.uploadGaussians(gaussians);
        }
    } catch (const std::exception& e) {
        cerr << "Failed to load PLY: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    
    GLuint colorTexture;
    glGenTextures(1, &colorTexture);
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, kWindowWidth, kWindowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        cerr << "Framebuffer not complete!" << '\n';
        return EXIT_FAILURE;
    }
    
    cudaRenderer.registerFramebuffer(colorTexture);
    
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    const char* vertexShaderSource = R"(
        #version 410 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";
    
    const char* fragmentShaderSource = R"(
        #version 410 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
    )";
    
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    GLuint displayProgram = glCreateProgram();
    glAttachShader(displayProgram, vertexShader);
    glAttachShader(displayProgram, fragmentShader);
    glLinkProgram(displayProgram);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // Store number of Gaussians for window title
    size_t numGaussians = gaussians.size();
    
    double lastTime = glfwGetTime();
    double fpsTimer = 0.0;
    int frameCount = 0;
    float avgFrameTime = 0.0f;
    float lastFPS = 0.0f;  // Store last calculated FPS for terminal output
    float terminalUpdateTimer = 0.0f;  // Timer for terminal output (every 5 seconds)
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !gMouseCaptured) {
            gMouseCaptured = true;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && gMouseCaptured) {
            gMouseCaptured = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        
        double now = glfwGetTime();
        float dt = static_cast<float>(now - lastTime);
        lastTime = now;
        
        if (gAutoBenchmark) {
            updateAutoBenchmarkCamera(gBenchmarkFrame);
            gBenchmarkFrame++;
            
            if (g_benchmark.isComplete()) {
                cout << "\nAuto-benchmark complete. Exiting." << endl;
                break;
            }
        } else {
            updateCameraPosition(window, dt);
        }
        
        frameCount++;
        fpsTimer += dt;
        terminalUpdateTimer += dt;
        
        // Update window title with FPS and number of Gaussians (once per second)
        if (fpsTimer >= 1.0) {
            avgFrameTime = (float)(fpsTimer / frameCount) * 1000.0f;
            lastFPS = frameCount / (float)fpsTimer;
            
            std::ostringstream title;
            title << "CUDA Renderer | " << numGaussians << " Gaussians | "
                  << std::fixed << std::setprecision(1) << lastFPS << " FPS | "
                  << std::setprecision(2) << avgFrameTime << " ms";
            
            if (gAutoBenchmark) {
                title << " | Benchmark: " << gBenchmarkFrame << "/" << (g_benchmark.config.warmupFrames + g_benchmark.config.measureFrames);
            }
            
            glfwSetWindowTitle(window, title.str().c_str());
            
            frameCount = 0;
            fpsTimer = 0.0;
        }
        
        // Print camera info and performance to terminal every 5 seconds
        if (terminalUpdateTimer >= 5.0f) {
            cout << "\n--- Camera & Performance Info ---" << endl;
            cout << "Gaussians: " << numGaussians << endl;
            if (lastFPS > 0.0f) {
                cout << "FPS: " << std::fixed << std::setprecision(1) << lastFPS << " | Frame Time: " << std::setprecision(2) << avgFrameTime << " ms" << endl;
            } else {
                // Calculate from current frames if we haven't had a full second yet
                float currentFPS = (frameCount > 0 && fpsTimer > 0) ? (frameCount / fpsTimer) : 0.0f;
                float currentTime = (frameCount > 0 && fpsTimer > 0) ? (fpsTimer / frameCount * 1000.0f) : 0.0f;
                cout << "FPS: " << std::fixed << std::setprecision(1) << currentFPS << " | Frame Time: " << std::setprecision(2) << currentTime << " ms" << endl;
            }
            cout << "Position: (" 
                 << std::fixed << std::setprecision(3) << gCamera.pos.x << ", "
                 << gCamera.pos.y << ", " << gCamera.pos.z << ")" << endl;
            cout << "Yaw: " << std::setprecision(1) << gCamera.yaw << "° | Pitch: " << gCamera.pitch << "° | FOV: " << gCamera.fovDeg << "°" << endl;
            cout << "Command to reuse: --pos \"" << gCamera.pos.x << "," << gCamera.pos.y << "," << gCamera.pos.z 
                 << "\" --yaw " << gCamera.yaw << " --pitch " << gCamera.pitch << " --fov " << gCamera.fovDeg << endl;
            
            terminalUpdateTimer = 0.0f;
        }
        
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
        
        vec3 forward{
            cos(radians(gCamera.yaw)) * cos(radians(gCamera.pitch)),
            sin(radians(gCamera.pitch)),
            sin(radians(gCamera.yaw)) * cos(radians(gCamera.pitch))
        };
        forward = normalize(forward);
        
        mat4 view = lookAt(gCamera.pos, gCamera.pos + forward, vec3(0.0f, 1.0f, 0.0f));
        float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
        mat4 projection = perspective(radians(gCamera.fovDeg), aspect, kNearPlane, kFarPlane);
        
        cudaRenderer.render(view, projection, gCamera);
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.07f, 0.07f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUseProgram(displayProgram);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        
        glfwSwapBuffers(window);
    }
    
    cudaRenderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return EXIT_SUCCESS;
}
