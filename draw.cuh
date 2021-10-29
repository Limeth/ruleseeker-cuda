#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <vector>
#include "shaders.cuh"

using namespace std;

// struktura bitmapy
typedef struct {
    int width;          // sirka bitmapy v pixelech
    int height;         // vyska bitmapy v pixelech
    uchar4 *pixels;     // ukazatel na bitmapu na strane CPU
    uchar4 *deviceData; // ukazatel na data bitmapy na GPU
} bitmap_t;

bitmap_t *bitmap = NULL;
u32 vao;
u32 vbo;
u32 shader_vertex;
u32 shader_fragment;
u32 shader_program;
vector<f32vec2> vertices;

// vykresleni bitmapy v OpenGL
__host__ void draw_image() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    if (bitmap != NULL && bitmap->pixels != NULL) {
        /* glDrawPixels(bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels); */

        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(f32vec2), &vertices[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(f32vec2), (void*) 0);

        glUseProgram(shader_program);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 3);
    }

    glutSwapBuffers();
}

__host__ u32 create_shader(u8* shader_code, i32 shader_len, GLenum shader_type) {
    u32 shader = glCreateShader(shader_type);
    u8* config_h_ptr = config_h;
    u8* shader_sources[3] {
        (u8*) "#version 450 core\n",
        config_h_ptr,
        shader_code,
    };
    GLint shader_source_lengths[3] = {
        18,
        (GLint) config_h_len,
        (GLint) shader_len,
    };
    glShaderSource(shader, 3, (const GLchar *const *) &shader_sources, (const GLint *) &shader_source_lengths);
    glCompileShader(shader);

    int  success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader_vertex, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

__host__ void init_draw(
        int argc,
        char **argv,
        int width,
        int height,
        void (handle_keys)(unsigned char key, int x, int y),
        void (idle_func)()
) {
    // alokace struktury bitmapy
    bitmap = (bitmap_t*) malloc(sizeof(bitmap));
    bitmap->width = GRID_WIDTH;
    bitmap->height = GRID_HEIGHT;

    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutCreateWindow("Life Game");
    glutDisplayFunc(draw_image);
    glutKeyboardFunc(handle_keys);
    glutIdleFunc(idle_func);

    glewInit();

    vertices.push_back(make_f32vec2(-1.0, -1.0));
    vertices.push_back(make_f32vec2( 1.0, -1.0));
    vertices.push_back(make_f32vec2( 0.0,  1.0));

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    shader_vertex = create_shader(shader_vert, shader_vert_len, GL_VERTEX_SHADER);
    shader_fragment = create_shader(shader_frag, shader_frag_len, GL_FRAGMENT_SHADER);

    shader_program = glCreateProgram();
    glAttachShader(shader_program, shader_vertex);
    glAttachShader(shader_program, shader_fragment);
    glLinkProgram(shader_program);

    int success;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);

    if(!success) {
        char infoLog[512];
        glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
        std::cout << "ERROR::PROGRAM::LINK_FAILED\n" << infoLog << std::endl;
    }

    glUseProgram(shader_program);
    glDeleteShader(shader_vertex);
    glDeleteShader(shader_fragment);
}

__host__ void finalize_draw() {
    free(bitmap);
}


int ui_loop() {
    glutMainLoop();

    return 0;
}
