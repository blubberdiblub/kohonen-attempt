#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT {
    mat4 orient;
    vec3 color;
} p[];

out vec3 v_color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void emit_vertex(vec4 pos, vec3 col) {
    gl_Position = projection * view * model * pos;
    v_color = col;
    EmitVertex();
}

void emit_particle(vec4 pos, float size, mat4 orient, vec3 col) {
    vec4 side = orient[0];
    vec4 up = orient[1];

    emit_vertex(pos + size * up,
                col * vec3(1.0, 0.0, 0.0));

    emit_vertex(pos + size * (up * cos(radians(120.0)) + side * sin(radians(120.0))),
                col * vec3(0.0, 1.0, 0.0));

    emit_vertex(pos + size * (up * cos(radians(240.0)) + side * sin(radians(240.0))),
                col * vec3(1.0, 1.0, 0.0));

    EndPrimitive();
}

void main() {
    emit_particle(gl_in[0].gl_Position,
                  gl_in[0].gl_PointSize,
                  p[0].orient,
                  p[0].color);
}
