#pragma once
#include <bsd/stdlib.h>
#include "util.cuh"
#include "math.cuh"

using namespace std;

#define STATES_TYPE_UNDEF 0
#define STATES_TYPE_OPENGL 1
#define STATES_TYPE_CUDA 2

typedef struct {
    GLuint gpu_vbo_grid_states_1;
    GLuint gpu_vbo_grid_states_2;
    struct cudaGraphicsResource* gpu_cuda_grid_states_1;
    struct cudaGraphicsResource* gpu_cuda_grid_states_2;
} gpu_states_opengl_t;

typedef struct {
    u8* gpu_cuda_grid_states_1;
    u8* gpu_cuda_grid_states_2;
} gpu_states_cuda_t;

typedef struct {
    u8 type;
    union {
        gpu_states_opengl_t opengl;
        gpu_states_cuda_t cuda;
    } gpu_states;
} gpu_states_t;

typedef struct {
    cudaStream_t stream;

    // gpu
    u8* gpu_ruleset;
    gpu_states_t gpu_states;
    u32* gpu_fit_cells_block_sums;

    // cpu
    u8* cpu_ruleset;
    u8* cpu_grid_states_1;
    u8* cpu_grid_states_2;
    u8* cpu_grid_states_tmp;
    u32* cpu_fit_cells_block_sums;
    u32 cpu_fit_cells;
    // value in the range of [0; 1] where 1 is most fit
    f32 fitness;
    // cumulatively summed errors derived from fitness in each iteration in which
    // fitness is computed.
    f32 cumulative_error;
} simulation_t;

// Capitalised because they are effectively constant
int CPU_CELL_NEIGHBOURHOOD_COMBINATIONS = -1;
__constant__ int GPU_CELL_NEIGHBOURHOOD_COMBINATIONS = -1;
int CPU_RULESET_SIZE = -1;
__constant__ int GPU_RULESET_SIZE = -1;

__inline__ __host__ __device__ int get_cell_neighbourhood_combinations() {
#ifdef __CUDA_ARCH__
    return GPU_CELL_NEIGHBOURHOOD_COMBINATIONS;
#else
    return CPU_CELL_NEIGHBOURHOOD_COMBINATIONS;
#endif
}

__inline__ __host__ __device__ int get_ruleset_size() {
#ifdef __CUDA_ARCH__
    return GPU_RULESET_SIZE;
#else
    return CPU_RULESET_SIZE;
#endif
}

void grid_init_random(u8* grid) {
    for (i32 y = 0; y < GRID_HEIGHT; y++) {
        for (i32 x = 0; x < GRID_WIDTH; x++) {
            i32 i = x + y * GRID_PITCH;
            grid[i] = (u8) random_sample_u32(CELL_STATES);
        }
    }
}

/**
 * Initializes a simulation with the provided stream.
 *
 * params:
 *   simulation: The simulation to initialize.
 *   opengl_interop: Whether OpenGL-interoperability should be enabled.
 *                   If `true`, respective OpenGL resources must already be initialized!
 *   provided_stream: The stream to use for computations. If 0, a new non-blocking stream is created.
 */
void simulation_init(simulation_t* simulation, bool opengl_interop, cudaStream_t provided_stream) {
    if (provided_stream) {
        simulation->stream = provided_stream;
    } else {
        CHECK_ERROR(cudaStreamCreateWithFlags(&simulation->stream, cudaStreamNonBlocking));
    }

    CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_ruleset), get_ruleset_size() * sizeof(u8)));
    CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_fit_cells_block_sums), GRID_AREA_IN_BLOCKS * sizeof(u32)));

    /* simulation->cpu_ruleset = (u8*) calloc(get_ruleset_size(), sizeof(u8)); */
    /* simulation->cpu_grid_states_1 = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8)); */
    /* simulation->cpu_grid_states_2 = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8)); */
    /* simulation->cpu_grid_states_tmp = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8)); */
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_ruleset), get_ruleset_size() * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_grid_states_1), GRID_AREA_WITH_PITCH * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_grid_states_2), GRID_AREA_WITH_PITCH * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_grid_states_tmp), GRID_AREA_WITH_PITCH * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_fit_cells_block_sums), GRID_AREA_IN_BLOCKS * sizeof(u32), simulation->stream));

    CHECK_ERROR(cudaStreamSynchronize(simulation->stream));

    // Initialize ruleset. Keep first rule as 0.
    for (i32 i = 1; i < get_ruleset_size(); i++) {
        simulation->cpu_ruleset[i] = (u8) random_sample_u32(CELL_STATES);
    }

    grid_init_random(simulation->cpu_grid_states_1);
    /* simulation->cpu_grid_states_1[(GRID_HEIGHT / 2) * GRID_PITCH + GRID_WIDTH / 2] = 1; */

    CHECK_ERROR(cudaStreamSynchronize(simulation->stream));

    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    if (opengl_interop) {
        simulation->gpu_states.type = STATES_TYPE_OPENGL;

        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->stream));
        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, simulation->stream));

        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_1, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1));
        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_2, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2));
    } else {
        simulation->gpu_states.type = STATES_TYPE_CUDA;

        CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1), GRID_AREA_WITH_PITCH * sizeof(u8)));
        CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2), GRID_AREA_WITH_PITCH * sizeof(u8)));

        gpu_grid_states_1 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1;
        gpu_grid_states_2 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2;
    }

    // prekopirovani pocatecniho stavu do GPU
    cudaMemcpyAsync(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream);
    cudaMemcpyAsync(gpu_grid_states_2, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream);

    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->stream));
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, simulation->stream));
    }

    CHECK_ERROR(cudaMemcpyAsync(simulation->gpu_ruleset, simulation->cpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream));
    CHECK_ERROR(cudaStreamSynchronize(simulation->stream));

    simulation->cpu_fit_cells = 0;
    simulation->fitness = 0.0f;
    simulation->cumulative_error = 0.0f;
}

void simulation_free(simulation_t* simulation) {
    if (simulation->gpu_states.type == STATES_TYPE_UNDEF) {
        return; // uninitialized
    } else if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        glDeleteBuffers(1, &simulation->gpu_states.gpu_states.opengl.gpu_vbo_grid_states_1);
        glDeleteBuffers(1, &simulation->gpu_states.gpu_states.opengl.gpu_vbo_grid_states_2);
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        cudaFree(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1);
        cudaFree(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2);
    }

    cudaFree(simulation->gpu_ruleset);
    cudaFree(simulation->gpu_fit_cells_block_sums);

    cudaFreeHost(simulation->cpu_ruleset);
    cudaFreeHost(simulation->cpu_grid_states_1);
    cudaFreeHost(simulation->cpu_grid_states_2);
    cudaFreeHost(simulation->cpu_grid_states_tmp);
    cudaFreeHost(simulation->cpu_fit_cells_block_sums);
}

void simulation_swap_buffers_gpu(simulation_t* simulation) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        swap(simulation->gpu_states.gpu_states.opengl.gpu_vbo_grid_states_1, simulation->gpu_states.gpu_states.opengl.gpu_vbo_grid_states_2);
        swap(simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2);
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        swap(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1, simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2);
    } else {
        fprintf(stderr, "Cannot swap buffers of an uninitialized simulation.\n");
        exit(1);
    }
}

void simulation_swap_buffers_cpu(simulation_t* simulation) {
    swap(simulation->cpu_grid_states_1, simulation->cpu_grid_states_2);
}

void simulation_gpu_states_map(simulation_t* simulation, u8** gpu_grid_states_1, u8** gpu_grid_states_2) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        if (gpu_grid_states_1) {
            CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->stream));
            CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) gpu_grid_states_1, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1));
        }

        if (gpu_grid_states_2) {
            CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, simulation->stream));
            CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) gpu_grid_states_2, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2));
        }
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        if (gpu_grid_states_1) {
            *gpu_grid_states_1 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1;
        }

        if (gpu_grid_states_2) {
            *gpu_grid_states_2 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2;
        }
    } else {
        fprintf(stderr, "Cannot map states of an uninitialized simulation.\n");
        exit(1);
    }
}

void simulation_gpu_states_unmap(simulation_t* simulation) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->stream));
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, simulation->stream));
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        // noop
    } else {
        fprintf(stderr, "Cannot unmap states of an uninitialized simulation.\n");
        exit(1);
    }
}

// Writes a ruleset to a file
void ruleset_save(char* filename, u8* ruleset) {
    if (!file_write(filename, ruleset, get_ruleset_size())) {
        fprintf(stderr, "Failed to write a ruleset.");
        exit(1);
    }
}

// Loads a ruleset from a file to a pre-allocated buffer
void ruleset_load(char* filename, u8* ruleset) {
    if (!file_load(filename, ruleset, get_ruleset_size())) {
        fprintf(stderr, "Failed to load a ruleset.");
        exit(1);
    }
}

void simulation_copy_grid_cpu_gpu(simulation_t* simulation) {
    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    cudaMemcpyAsync(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream);
    simulation_gpu_states_unmap(simulation);
}

void simulation_copy_ruleset_gpu_cpu(simulation_t* simulation) {
    CHECK_ERROR(cudaMemcpyAsync(simulation->cpu_ruleset, simulation->gpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyDeviceToHost, simulation->stream));
}

void simulation_copy_ruleset_cpu_gpu(simulation_t* simulation) {
    CHECK_ERROR(cudaMemcpyAsync(simulation->gpu_ruleset, simulation->cpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream));
}

void simulation_ruleset_save(simulation_t* simulation, char* filename) {
    simulation_copy_ruleset_gpu_cpu(simulation);
    cudaStreamSynchronize(simulation->stream);
    ruleset_save(filename, simulation->cpu_ruleset);
}

void simulation_ruleset_load(simulation_t* simulation, char* filename) {
    ruleset_load(filename, simulation->cpu_ruleset);
    CHECK_ERROR(cudaMemcpyAsync(simulation->gpu_ruleset, simulation->cpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream));
    cudaStreamSynchronize(simulation->stream);
}

void grid_save(char* filename, u8* grid) {
    if (!file_write_2d_pitch(filename, grid, GRID_WIDTH, GRID_HEIGHT, GRID_PITCH)) {
        fprintf(stderr, "Failed to write a grid.");
        exit(1);
    }
}

void grid_load(char* filename, u8* grid) {
    if (!file_load_2d_pitch(filename, grid, GRID_WIDTH, GRID_HEIGHT, GRID_PITCH)) {
        fprintf(stderr, "Failed to load a grid.");
        exit(1);
    }
}

void simulation_grid_save(simulation_t* simulation, char* filename) {
    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    CHECK_ERROR(cudaMemcpyAsync(simulation->cpu_grid_states_1, gpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyDeviceToHost, simulation->stream));
    cudaStreamSynchronize(simulation->stream);
    simulation_gpu_states_unmap(simulation);
    grid_save(filename, simulation->cpu_grid_states_1);
}

void simulation_grid_load(simulation_t* simulation, char* filename) {
    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    grid_load(filename, simulation->cpu_grid_states_1);
    CHECK_ERROR(cudaMemcpyAsync(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream));
    cudaStreamSynchronize(simulation->stream);
    simulation_gpu_states_unmap(simulation);
}

void simulation_collect_fit_cells_block_sums_async(simulation_t* simulation) {
    cudaMemcpyAsync(simulation->cpu_fit_cells_block_sums, simulation->gpu_fit_cells_block_sums, GRID_AREA_IN_BLOCKS * sizeof(u32), cudaMemcpyDeviceToHost, simulation->stream);
}

void simulation_reduce_fit_cells(simulation_t* simulation) {
    simulation->cpu_fit_cells = 0;

    for (i32 i = 0; i < GRID_AREA_IN_BLOCKS; i++) {
        simulation->cpu_fit_cells += simulation->cpu_fit_cells_block_sums[i];
    }
}

void simulation_compute_fitness(simulation_t* simulation) {
    f32 proportion_actual = ((f32) simulation->cpu_fit_cells) / ((f32) GRID_AREA);
    f32 fitness;

    // See https://www.desmos.com/calculator/b8daos2dqt
    f32 x = proportion_actual; // aliases
    const f32 p = FITNESS_FN_TARGET;

    if (FITNESS_FN_TYPE == FITNESS_FN_TYPE_ABS) {
        fitness = abs(x - p);
    } else if (FITNESS_FN_TYPE == FITNESS_FN_TYPE_LINEAR) {
        fitness = min(x / p, (1.0f - x) / (1.0f - p));
    } else if (FITNESS_FN_TYPE == FITNESS_FN_TYPE_LIKELIHOOD) {
        // normalize to fit values exactly in the range [0; 1]
        const f32 normalization_factor = 1.0f / (powf(1.0f - p, 1.0f - p) * powf(p, p));
        fitness = normalization_factor * powf(1.0f - x, 1.0f - p) * powf(x, p);
    } else {
        printf("Invalid fitness function.\n");
        exit(1);
    }

    simulation->fitness = fitness;
}

void simulation_cumulative_error_reset(simulation_t* simulation) {
    simulation->cumulative_error = 0.0f;
}

void simulation_cumulative_error_add(simulation_t* simulation) {
    f32 error_sqrt = 1.0f - simulation->fitness;
    simulation->cumulative_error += error_sqrt * error_sqrt;
}

void simulation_cumulative_error_normalize(simulation_t* simulation) {
    simulation->cumulative_error /= (f32) FITNESS_EVAL_LEN;
}

void ruleset_crossover(u8* a, u8* b, u8* target) {
    u32 ruleset_size = get_ruleset_size();

    for (u32 i = 0; i < ruleset_size; i++) {
        if (random_sample_u32(2)) {
            target[i] = a[i];
        } else {
            target[i] = b[i];
        }
    }
}

void ruleset_mutate(u8* ruleset) {
    u32 ruleset_size = get_ruleset_size();

    // There is probably a more efficient way to mutate the genome, maybe using the central limit theorem
    for (u32 i = 0; i < ruleset_size; i++) {
        if (random_sample_f32_normalized() < MUTATION_CHANCE) {
            ruleset[i] = (u8) random_sample_u32(CELL_STATES);
        }
    }
}
