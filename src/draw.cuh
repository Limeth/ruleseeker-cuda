#pragma once
#include "util.cuh"
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <time.h>
#include <png.h>
#include "shaders.cuh"
#include "simulation.cuh"

using namespace std;

void (*global_idle_func)();
void (*global_window_close_callback)(GLFWwindow*);

GLFWwindow* window;

int global_argc;
char** global_argv;
bool editing_mode = false;;
bool randomize_grid = false;;
u32 pressed_cell_index = (u32) -1;

GLuint fbo;
GLuint texture_color;
GLuint texture_cells;
GLuint vao;
GLuint vbo;
GLuint shader_vertex;
GLuint shader_fragment;
GLuint shader_program;
GLuint uniform_window_resolution;

simulation_t preview_simulation;

/* GLuint gpu_vbo_grid_states_1; */
/* GLuint gpu_vbo_grid_states_2; */

/* struct cudaGraphicsResource* gpu_cuda_grid_states_1 = NULL; */
/* struct cudaGraphicsResource* gpu_cuda_grid_states_2 = NULL; */

u32 frame_index = 0;
u32vec2 window_size = make_u32vec2(WINDOW_WIDTH, WINDOW_HEIGHT);

#ifdef EXPORT_FRAMES
__host__ void export_frame() {
    if (frame_index >= EXPORT_FRAMES) {
        return;
    }

    u8 buffer[window_size.x * window_size.y * 4 * sizeof(u8)];
    glReadPixels(0, 0, window_size.x, window_size.y, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

    char* file_name;

    if (asprintf(&file_name, "frame_%04u.png", frame_index) == -1) {
        fprintf(stderr, "Could not format a string.\n");
        exit(1);
    }

    png_image image;
    image.version = PNG_IMAGE_VERSION;
    image.width = window_size.x;
    image.height = window_size.y;
    image.format = PNG_FORMAT_RGBA;

    png_image_write_to_file(&image, file_name, false, buffer, window_size.x * 4 * sizeof(u8), NULL);

    free(file_name);

    if (frame_index + 1 >= EXPORT_FRAMES && EXPORT_FRAMES > 0) {
        printf("Done exporting frames.\n");
    }
}
#endif

__host__ void draw_image() {
    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glNamedFramebufferDrawBuffers(fbo, 2, bufs);

    u8 clear_color[4] = { 0, 0, 0, 0 };
    glClearTexImage(texture_color, 0, GL_RGBA, GL_BYTE, clear_color);

    i32 clear_cell[1] = { -1 };
    glClearTexImage(texture_cells, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, clear_cell);

    // Draw
    {
        glBindVertexArray(vao);

        /* glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(f32vec2), &vertices[0], GL_STATIC_DRAW); */

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, preview_simulation.gpu_states.gpu_states.opengl.gpu_vbo_grid_states_1);
        glVertexAttribIPointer(0, 1, GL_UNSIGNED_BYTE, sizeof(u8), (void*) 0);
        glVertexAttribDivisor(0, 1);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, preview_simulation.gpu_states.gpu_states.opengl.gpu_vbo_grid_states_2);
        glVertexAttribIPointer(1, 1, GL_UNSIGNED_BYTE, sizeof(u8), (void*) 0);
        glVertexAttribDivisor(1, 1);

        glUseProgram(shader_program);

        glBindVertexArray(vao);

        // render each row separately, so that the pitch gap between aligned rows is skipped
        for (i32 y = 0; y < GRID_HEIGHT; y++) {
            glDrawArraysInstancedBaseInstance(GL_TRIANGLE_FAN, 0, CELL_VERTICES, GRID_WIDTH, y * GRID_PITCH);
        }
    }
}

// vykresleni bitmapy v OpenGL
__host__ void display_func() {
    draw_image();

#ifdef EXPORT_FRAMES
    if (!editing_mode) {
        export_frame();
    }
#endif

    glBlitNamedFramebuffer(fbo, 0, 0, 0, window_size.x, window_size.y, 0, 0, window_size.x, window_size.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glfwSwapBuffers(window);

#ifdef SLEEP_MS
    msleep(SLEEP_MS);
#endif

#ifdef EXIT_AFTER_FRAMES
    if (!editing_mode) {
        if (frame_index + 1 >= EXIT_AFTER_FRAMES) {
            printf("Reached max number of frames, exiting.\n");
            exit(0);
        }
    }
#endif

    frame_index += 1;
}

__host__ u32 create_shader(u8* shader_code, i32 shader_len, GLenum shader_type) {
    u32 shader = glCreateShader(shader_type);
    u8* config_h_ptr = config_h;
    u8* config_derived_h_ptr = config_derived_h;
    u8* shader_sources[4] {
        (u8*) "#version 460 core\n",
        config_h_ptr,
        config_derived_h_ptr,
        shader_code,
    };
    GLint shader_source_lengths[4] = {
        18,
        (GLint) config_h_len,
        (GLint) config_derived_h_len,
        (GLint) shader_len,
    };
    glShaderSource(shader, 4, (const GLchar *const *) &shader_sources, (const GLint *) &shader_source_lengths);
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

void update_window_size_dependent_resources(int width, int height) {
    printf("Window size changed: %d, %d\n", width, height);
    window_size.x = width;
    window_size.y = height;
    glViewport(0, 0, width, height);
    glUniform2ui(uniform_window_resolution, width, height);

    u32 texture_cells_data[width][height];

    glDeleteTextures(1, &texture_color);
    glGenTextures(1, &texture_color);
    glBindTexture(GL_TEXTURE_2D, texture_color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_color, 0);

    glDeleteTextures(1, &texture_cells);
    glGenTextures(1, &texture_cells);
    glBindTexture(GL_TEXTURE_2D, texture_cells);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, texture_cells_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, texture_cells, 0);
}

void window_size_callback(GLFWwindow* window, int width, int height) {
    update_window_size_dependent_resources(width, height);
}

void handle_mouse(GLFWwindow* window, int button, int action, int mods) {
    if (!editing_mode) {
        return;
    }

    double mouse_x, mouse_y;

    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            i32 pixel_x = mouse_x;
            i32 pixel_y = window_size.y - 1 - ((i32) mouse_y);

            glReadBuffer(GL_COLOR_ATTACHMENT1);
            glReadPixels(pixel_x, pixel_y, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &pressed_cell_index);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
        } else if (action == GLFW_RELEASE) {
            if (pressed_cell_index != (u32) -1) {
                u8 state = preview_simulation.cpu_grid_states_1[pressed_cell_index];

                if (button == GLFW_MOUSE_BUTTON_LEFT) {
                    state = (state + 1) % CELL_STATES;
                } else {
                    state = (state + CELL_STATES - 1) % CELL_STATES;
                }

                preview_simulation.cpu_grid_states_1[pressed_cell_index] = state;

                simulation_copy_grid_cpu_gpu(&preview_simulation);
            }

            pressed_cell_index = (u32) -1;
        }
    }
}

void handle_keys(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
        glfwSetWindowShouldClose(window, true);
        global_window_close_callback(window);
    }
}

__host__ void init_draw(
        int argc,
        char **argv,
        void (window_close_callback)(GLFWwindow* window),
        void (param_idle_func)(),
        bool edit
) {
    editing_mode = edit;

    if (editing_mode) {
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-r") == 0) {
                randomize_grid = true;
            }
        }
    }

    if (!glfwInit()) {
        exit(1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    /* glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE); */

    if (USE_MULTISAMPLING) {
        glfwWindowHint(GLFW_SAMPLES, MULTISAMPLING_SAMPLES);
    }

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "ruleseeker-cuda", NULL, NULL);

    if (!window) {
        glfwTerminate();
        exit(1);
    }

    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, handle_keys);
    glfwSetMouseButtonCallback(window, handle_mouse);
    glfwSetWindowCloseCallback(window, window_close_callback);
    glfwMakeContextCurrent(window);

    global_idle_func = param_idle_func;
    global_window_close_callback = window_close_callback;

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(gl_debug_message_callback, NULL);
    /* glEnable(GL_FRAMEBUFFER_SRGB); */
    glDisable(GL_DEPTH_TEST);

    if (USE_MULTISAMPLING) {
        glEnable(GL_MULTISAMPLE);
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    preview_simulation.gpu_states.type = STATES_TYPE_OPENGL;

    create_cuda_vbo(&preview_simulation.gpu_states.gpu_states.opengl.gpu_vbo_grid_states_1, &preview_simulation.gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaGraphicsRegisterFlagsNone);
    create_cuda_vbo(&preview_simulation.gpu_states.gpu_states.opengl.gpu_vbo_grid_states_2, &preview_simulation.gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, GRID_AREA_WITH_PITCH * sizeof(u8), cudaGraphicsRegisterFlagsNone);

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
    if (editing_mode) {
        printf("Saving grid to: %s\n", global_argv[2]);
        simulation_grid_save(&preview_simulation, global_argv[2]);
    }
}


int ui_loop() {
    // Manually update window size dependent resources as the GLFW resize callback is not called on launch.
    update_window_size_dependent_resources(WINDOW_WIDTH, WINDOW_HEIGHT);

    bool first = true;

    while (!glfwWindowShouldClose(window)) {
        if (first) {
            first = false;
        } else {
            global_idle_func();
        }

        display_func();

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}
