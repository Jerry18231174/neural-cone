#version 460 core

out vec4 FragColor;

in vec2 texCoords;

layout(location = 0, binding = 0) uniform sampler2D Image;
layout(location = 1) uniform vec4 v1;
#define g_use_tonemapping (v1.x != 0)
#define g_exposure v1.y

vec3 tonemap_aces(vec3 value) {
    // float A = 2.51;
    // float B = 0.03;
    // float C = 2.43;
    // float D = 0.59;
    // float E = 0.14;
    // value *= 0.6;
    // value = clamp((value * (A * value + B)) / (value * (C * value + D) + E), 0.0, 1.0);
    value *= g_exposure;
    value = log(value + vec3(1.0));
    value = pow(value, vec3(0.45454545));  // gamma 2.2
    return value;
}

void main() {
    vec3 color = texture(Image, texCoords).xyz;
    if (g_use_tonemapping) {
        color = tonemap_aces(color);
    }
    FragColor = vec4(color, 1.0);
}