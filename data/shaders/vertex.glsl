#version 330

in vec3 in_vert;
in mat4 in_orient;
in vec3 in_color;

out VS_OUT {
    mat4 orient;
    vec3 color;
} p;

void main() {
    p.orient = in_orient;
    p.color = in_color;
    gl_Position = vec4(in_vert, 1.0);
    gl_PointSize = 0.04;
}
