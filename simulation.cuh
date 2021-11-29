#pragma once
#include "util.cuh"
#include "math.cuh"

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
    // gpu
    u8* gpu_ruleset;
    gpu_states_t gpu_states;
    u32* gpu_fitness_sums;

    // cpu
    u8* cpu_ruleset;
    u8* cpu_grid_states_1;
    u8* cpu_grid_states_2;
    u8* cpu_grid_states_tmp;
    u32* cpu_fitness_sums;
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

void simulation_gpu_states_map(simulation_t* simulation, u8** gpu_grid_states_1, u8** gpu_grid_states_2) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, 0));
        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, 0));
        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) gpu_grid_states_1, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1));
        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) gpu_grid_states_2, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2));
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        *gpu_grid_states_1 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1;
        *gpu_grid_states_2 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2;
    } else {
        fprintf(stderr, "Cannot map states of an uninitialized simulation.\n");
        exit(1);
    }
}

void simulation_gpu_states_unmap(simulation_t* simulation) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, 0));
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, 0));
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        // noop
    } else {
        fprintf(stderr, "Cannot unmap states of an uninitialized simulation.\n");
        exit(1);
    }
}

void simulation_init(simulation_t* simulation) {
    // alokovani mista pro bitmapu na GPU
    CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_ruleset), get_ruleset_size() * sizeof(u8)));

    simulation->cpu_ruleset = (u8*) calloc(get_ruleset_size(), sizeof(u8));
    simulation->cpu_grid_states_1 = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8));
    simulation->cpu_grid_states_2 = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8));
    simulation->cpu_grid_states_tmp = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8));

    /* for (int i = 0; i < GRID_AREA; i++) { */
    /*     cpu_grid_states_1[i] = (u8) (rand() % CELL_STATES); */
    /* } */
    simulation->cpu_grid_states_1[(GRID_HEIGHT / 2) * GRID_PITCH + GRID_WIDTH / 2] = 1;

    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, 0));
        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, 0));

        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_1, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1));
        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_2, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2));
    } else {
        simulation->gpu_states.type = STATES_TYPE_CUDA;

        CHECK_ERROR(cudaMalloc((void**) &simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8)));
        CHECK_ERROR(cudaMalloc((void**) &simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2, GRID_AREA_WITH_PITCH * sizeof(u8)));

        gpu_grid_states_1 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1;
        gpu_grid_states_2 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2;
    }

    // prekopirovani pocatecniho stavu do GPU
    cudaMemcpy(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_grid_states_2, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice);

    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, 0));
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, 0));
    }

    // Initialize ruleset. Keep first rule as 0.
    for (i32 i = 1; i < get_ruleset_size(); i++) {
        simulation->cpu_ruleset[i] = (u8) (rand() % CELL_STATES);
    }

    CHECK_ERROR(cudaMemcpy(simulation->gpu_ruleset, simulation->cpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyHostToDevice));
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
    cudaFree(simulation->gpu_fitness_sums);

    free(simulation->cpu_ruleset);
    free(simulation->cpu_grid_states_1);
    free(simulation->cpu_grid_states_2);
    free(simulation->cpu_grid_states_tmp);
    free(simulation->cpu_fitness_sums);
}
