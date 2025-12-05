#version 410 core

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexcoord;

out vec2 vUv;

void main() {
    vUv = aTexcoord;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}