#pragma once
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <time.h>
#include "shaders.cuh"

using namespace std;

GLuint vao;
GLuint vbo;
GLuint shader_vertex;
GLuint shader_fragment;
GLuint shader_program;
GLuint uniform_window_resolution;

GLuint gpu_vbo_grid_states_1;
GLuint gpu_vbo_grid_states_2;

struct cudaGraphicsResource* gpu_cuda_grid_states_1 = NULL;
struct cudaGraphicsResource* gpu_cuda_grid_states_2 = NULL;

// vykresleni bitmapy v OpenGL
__host__ void draw_image() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw
    {
        glBindVertexArray(vao);

        /* glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(f32vec2), &vertices[0], GL_STATIC_DRAW); */

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, gpu_vbo_grid_states_1);
        glVertexAttribIPointer(0, 1, GL_UNSIGNED_BYTE, sizeof(u8), (void*) 0);
        glVertexAttribDivisor(0, 1);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, gpu_vbo_grid_states_2);
        glVertexAttribIPointer(1, 1, GL_UNSIGNED_BYTE, sizeof(u8), (void*) 0);
        glVertexAttribDivisor(1, 1);

        glUseProgram(shader_program);

        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, CELL_VERTICES, GRID_AREA);
    }

    glutSwapBuffers();

#ifdef SLEEP_MS
    msleep(SLEEP_MS);
#endif
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
        GLsizei info_log_len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);
        GLchar info_log[info_log_len];
        glGetShaderInfoLog(shader, info_log_len, NULL, info_log);
        printf("ERROR::SHADER::COMPILATION_FAILED: %s\n", info_log);
        exit(1);
    }

    return shader;
}

// Create an OpenGL buffer accessible from CUDA
__host__ void create_cuda_vbo(GLuint *vbo, struct cudaGraphicsResource **vbo_res, u32 size, u32 vbo_res_flags) {
    /* assert(vbo); */

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    CHECK_ERROR(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

// Delete an OpenGL buffer accessible from CUDA
__host__ void delete_cuda_vbo(GLuint *vbo, struct cudaGraphicsResource *vbo_res) {
    // unregister this buffer object with CUDA
    CHECK_ERROR(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

void reshape_func(int width, int height) {
    printf("Window size changed: %d, %d\n", width, height);
    glViewport(0, 0, width, height);
    glUniform2ui(uniform_window_resolution, width, height);
}

__host__ void init_draw(
        int argc,
        char **argv,
        int width,
        int height,
        void (handle_keys)(unsigned char key, int x, int y),
        void (idle_func)()
) {
    glutInit(&argc, argv);

    u32 display_mode = GLUT_DOUBLE | GLUT_RGBA;

    if (USE_MULTISAMPLING) {
        display_mode |= GLUT_MULTISAMPLE;
        glutSetOption(GLUT_MULTISAMPLE, MULTISAMPLING_SAMPLES);
    }

    glutInitWindowSize(800, 800);
    glutInitDisplayMode(display_mode);
    glutCreateWindow("Life Game");
    glutDisplayFunc(draw_image);
    glutKeyboardFunc(handle_keys);
    glutIdleFunc(idle_func);
    glutReshapeFunc(reshape_func);

    glewInit();

    glDisable(GL_DEPTH_TEST);

    if (USE_MULTISAMPLING) {
        glEnable(GL_MULTISAMPLE);
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    create_cuda_vbo(&gpu_vbo_grid_states_1, &gpu_cuda_grid_states_1, GRID_AREA * sizeof(u8), cudaGraphicsRegisterFlagsNone);
    create_cuda_vbo(&gpu_vbo_grid_states_2, &gpu_cuda_grid_states_2, GRID_AREA * sizeof(u8), cudaGraphicsRegisterFlagsNone);

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
        exit(1);
    }

    uniform_window_resolution = glGetUniformLocation(shader_program, "window_resolution");

    glUseProgram(shader_program);
    glDeleteShader(shader_vertex);
    glDeleteShader(shader_fragment);
}

__host__ void finalize_draw() {
}


int ui_loop() {
    glutMainLoop();

    return 0;
}
