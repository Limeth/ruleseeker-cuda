#pragma once

/**
 * @file
 * @brief The main type `simulation_t` and its management.
 */

#include <bsd/stdlib.h>
#include <cub/cub.cuh>
#include "util.cuh"
#include "math.cuh"

using namespace std;

/**
 * @name Grid GPU allocation types
 * @{
 * The method the grid states are allocated on the GPU.
 */
/// Undefined.
#define STATES_TYPE_UNDEF 0
/// Allocated using OpenGL.
#define STATES_TYPE_OPENGL 1
/// Allocated using CUDA.
#define STATES_TYPE_CUDA 2
///@}

/// OpenGL-allocated grid buffers and their CUDA-mapped state.
typedef struct {
    /// OpenGL handle to the VBO of the first grid.
    GLuint gpu_vbo_grid_states_1;
    /// OpenGL handle to the VBO of the second grid.
    GLuint gpu_vbo_grid_states_2;
    /// Mapped CUDA resource corresponding to the first grid.
    struct cudaGraphicsResource* gpu_cuda_grid_states_1;
    /// Mapped CUDA resource corresponding to the second grid.
    struct cudaGraphicsResource* gpu_cuda_grid_states_2;
    /// Whether the first grid is mapped and therefore should be unmapped before rendering.
    bool gpu_cuda_grid_states_1_mapped;
    /// Whether the second grid is mapped and therefore should be unmapped before rendering.
    bool gpu_cuda_grid_states_2_mapped;
} gpu_states_opengl_t;

/// CUDA-allocated grid buffers. No mapping state necessary.
typedef struct {
    /// The CUDA-allocated buffer for the first grid.
    u8* gpu_cuda_grid_states_1;
    /// The CUDA-allocated buffer for the second grid.
    u8* gpu_cuda_grid_states_2;
} gpu_states_cuda_t;

/// A pair GPU-allocated grid buffers. One is used to compute the other.
typedef struct {
    u8 type;
    union {
        gpu_states_opengl_t opengl;
        gpu_states_cuda_t cuda;
    } gpu_states;
} gpu_states_t;

/// A simulation of a 2D cellular automaton.
/**
 *  Contains the current state of the grid and the ruleset used to compute the following iterations.
 */
typedef struct {
    /// The stream which to use to perform all CUDA commands.
    cudaStream_t stream;

    /**
     *  @name GPU
     *  @{
     */
    /// The ruleset on the GPU.
    u8* gpu_ruleset;
    /// The pair of grids on the GPU.
    gpu_states_t gpu_states;
    /// The GPU-allocated array of fit cells, one sum per block.
    u32* gpu_fit_cells_block_sums;
    /// The GPU-allocated total number of fit cells. Reduced from `gpu_fit_cells_block_sums`.
    u32* gpu_fit_cells;
    ///@}

    /**
     *  @name CPU
     *  @{
     */
    /// CUB temporary storage.
    temp_storage_t temp_storage;
    /// The ruleset on the CPU.
    u8* cpu_ruleset;
    /// The first grid of states on the CPU, used during verification.
    u8* cpu_grid_states_1;
    /// The second grid of states on the CPU, used during verification.
    u8* cpu_grid_states_2;
    /// The "temporary" grid of states on the CPU, used during verification.
    u8* cpu_grid_states_tmp;
    /// The CPU-allocated total number of fit cells. Downloaded from `gpu_fit_cells`.
    u32 cpu_fit_cells;
    /// Value in the range of [0; 1] where 1 is most fit. Computed from `cpu_fit_cells`.
    f32 fitness;
    /// Cumulatively summed errors derived from fitness in each iteration in which fitness is computed.
    f32 cumulative_error;
    ///@}
} simulation_t;

void grid_save(char* filename, u8* grid);
void grid_load(char* filename, u8* grid);

// Capitalised because they are effectively constant
/// Number of possible combinations of neighbouring states, on the CPU.
int CPU_CELL_NEIGHBOURHOOD_COMBINATIONS = -1;
/// Number of possible combinations of neighbouring states, on the GPU.
__constant__ int GPU_CELL_NEIGHBOURHOOD_COMBINATIONS = -1;
/// Number of rules in a ruleset, on the CPU.
int CPU_RULESET_SIZE = -1;
/// Number of rules in a ruleset, on the GPU.
__constant__ int GPU_RULESET_SIZE = -1;

/// Number of possible combinations of neighbouring states. See `compute_neighbouring_state_combinations`.
__inline__ __host__ __device__ int get_cell_neighbourhood_combinations() {
#ifdef __CUDA_ARCH__
    return GPU_CELL_NEIGHBOURHOOD_COMBINATIONS;
#else
    return CPU_CELL_NEIGHBOURHOOD_COMBINATIONS;
#endif
}

/// Number of rules in a ruleset. See `compute_ruleset_size`.
__inline__ __host__ __device__ int get_ruleset_size() {
#ifdef __CUDA_ARCH__
    return GPU_RULESET_SIZE;
#else
    return CPU_RULESET_SIZE;
#endif
}

/// Computes the number of iterations in which the fitness is computed.
int get_fitness_iterations() {
    int iterations[] = { FITNESS_FN_TARGET_ITERATIONS };
    int len = sizeof(iterations) / sizeof(int);

    return len;
}

/// Returns the required number of iterations to accumulate cumulative error.
int get_fitness_eval_to() {
    int iterations[] = { FITNESS_FN_TARGET_ITERATIONS };
    int len = sizeof(iterations) / sizeof(int);
    int max = 0;

    for (int i = 0; i < len; i++) {
        if (iterations[i] > max) {
            max = iterations[i];
        }
    }

    return max + 1;
}

/// Returns the target value of the fitness function.
float get_fitness_fn_target(int iteration) {
    int iterations[] = { FITNESS_FN_TARGET_ITERATIONS };
    float values[] = { FITNESS_FN_TARGET_VALUES };
    int len = sizeof(iterations) / sizeof(int);

    for (int i = 0; i < len; i++) {
        if (iteration == iterations[i]) {
            return values[i];
        }
    }

    return 0.0 / 0.0; // NaN
}

/// Whether or not the sum of fit cells should be performed for the given iteration.
bool get_fitness_fn_reduce(int iteration) {
    float target = get_fitness_fn_target(iteration);
    bool is_nan = target != target;

    return !is_nan;
}

/// Randomly initialize the CPU-allocated grid.
void grid_init_random(u8* grid) {
    for (i32 y = 0; y < GRID_HEIGHT; y++) {
        for (i32 x = 0; x < GRID_WIDTH; x++) {
            i32 i = x + y * GRID_PITCH;
            grid[i] = (u8) random_sample_u32(CELL_STATES);
        }
    }
}

/// Initializes a simulation with the provided stream.
/**
 * @param simulation: The simulation to initialize.
 * @param opengl_interop: Whether OpenGL-interoperability should be enabled. If `true`, respective OpenGL resources must already be initialized!
 * @param randomize_ruleset_cpu: Whether the ruleset should be randomized on the CPU.
 * @param grid_file: Not `NULL`, if the inintial grid state should be loaded from the file at the specified path.
 * @param randomize_grid: Whether the initial grid state should be randomized.
 * @param provided_stream: The stream to use for computations. If 0, a new non-blocking stream is created.
 */
void simulation_init(simulation_t* simulation, bool opengl_interop, bool randomize_ruleset_cpu, char* grid_file, bool randomize_grid, cudaStream_t provided_stream) {
    if (provided_stream) {
        simulation->stream = provided_stream;
    } else {
        CHECK_ERROR(cudaStreamCreateWithFlags(&simulation->stream, cudaStreamNonBlocking));
    }

    temp_storage_init(&simulation->temp_storage);

    CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_ruleset), get_ruleset_size() * sizeof(u8)));
    CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_fit_cells_block_sums), GRID_AREA_IN_BLOCKS * sizeof(u32)));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->gpu_fit_cells), sizeof(u32), simulation->stream));

    /* simulation->cpu_ruleset = (u8*) calloc(get_ruleset_size(), sizeof(u8)); */
    /* simulation->cpu_grid_states_1 = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8)); */
    /* simulation->cpu_grid_states_2 = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8)); */
    /* simulation->cpu_grid_states_tmp = (u8*) calloc(GRID_AREA_WITH_PITCH, sizeof(u8)); */
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_ruleset), get_ruleset_size() * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_grid_states_1), GRID_AREA_WITH_PITCH * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_grid_states_2), GRID_AREA_WITH_PITCH * sizeof(u8), simulation->stream));
    CHECK_ERROR(cudaCallocHostAsync((void**) &(simulation->cpu_grid_states_tmp), GRID_AREA_WITH_PITCH * sizeof(u8), simulation->stream));

    CHECK_ERROR(cudaStreamSynchronize(simulation->stream));

    // Initialize ruleset. Keep first rule as 0.
    if (randomize_ruleset_cpu) {
        for (i32 i = 1; i < get_ruleset_size(); i++) {
            simulation->cpu_ruleset[i] = (u8) random_sample_u32(CELL_STATES);
        }
    } else {
        memset(simulation->cpu_ruleset, 0, get_ruleset_size());
    }

    if (grid_file) {
        grid_load(grid_file, simulation->cpu_grid_states_1);
    } else if (randomize_grid) {
        grid_init_random(simulation->cpu_grid_states_1);
    } else {
        // All cells initialized to state 0.
    }

    CHECK_ERROR(cudaStreamSynchronize(simulation->stream));

    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    if (opengl_interop) {
        simulation->gpu_states.type = STATES_TYPE_OPENGL;

        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->stream));
        CHECK_ERROR(cudaGraphicsMapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, simulation->stream));

        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_1, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1));
        CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_2, NULL, simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2));

        simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1_mapped = true;
        simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2_mapped = true;
    } else {
        simulation->gpu_states.type = STATES_TYPE_CUDA;

        CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1), GRID_AREA_WITH_PITCH * sizeof(u8)));
        CHECK_ERROR(cudaMalloc((void**) &(simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2), GRID_AREA_WITH_PITCH * sizeof(u8)));

        gpu_grid_states_1 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_1;
        gpu_grid_states_2 = simulation->gpu_states.gpu_states.cuda.gpu_cuda_grid_states_2;
    }

    cudaMemcpyAsync(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream);
    cudaMemcpyAsync(gpu_grid_states_2, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream);

    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1, simulation->stream));
        CHECK_ERROR(cudaGraphicsUnmapResources(1, &simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2, simulation->stream));
        simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_1_mapped = false;
        simulation->gpu_states.gpu_states.opengl.gpu_cuda_grid_states_2_mapped = false;
    }

    CHECK_ERROR(cudaMemcpyAsync(simulation->gpu_ruleset, simulation->cpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream));
    CHECK_ERROR(cudaStreamSynchronize(simulation->stream));

    simulation->cpu_fit_cells = 0;
    simulation->fitness = 0.0f;
    simulation->cumulative_error = 0.0f;
}

/// Destructor for `simulation_t`.
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
}

/// Swap GPU grid state buffers, on the CPU.
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

/// Swap CPU grid state buffers.
void simulation_swap_buffers_cpu(simulation_t* simulation) {
    swap(simulation->cpu_grid_states_1, simulation->cpu_grid_states_2);
}

/// Map grid state buffers to ensure accessiblity from CUDA.
void simulation_gpu_states_map(simulation_t* simulation, u8** gpu_grid_states_1, u8** gpu_grid_states_2) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        gpu_states_opengl_t* opengl = &simulation->gpu_states.gpu_states.opengl;

        if (gpu_grid_states_1 != NULL && !opengl->gpu_cuda_grid_states_1_mapped) {
            CHECK_ERROR(cudaGraphicsMapResources(1, &opengl->gpu_cuda_grid_states_1, simulation->stream));
            CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) gpu_grid_states_1, NULL, opengl->gpu_cuda_grid_states_1));
            opengl->gpu_cuda_grid_states_1_mapped = true;
        }

        if (gpu_grid_states_2 != NULL && !opengl->gpu_cuda_grid_states_2_mapped) {
            CHECK_ERROR(cudaGraphicsMapResources(1, &opengl->gpu_cuda_grid_states_2, simulation->stream));
            CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) gpu_grid_states_2, NULL, opengl->gpu_cuda_grid_states_2));
            opengl->gpu_cuda_grid_states_2_mapped = true;
        }
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        gpu_states_cuda_t* cuda = &simulation->gpu_states.gpu_states.cuda;

        if (gpu_grid_states_1) {
            *gpu_grid_states_1 = cuda->gpu_cuda_grid_states_1;
        }

        if (gpu_grid_states_2) {
            *gpu_grid_states_2 = cuda->gpu_cuda_grid_states_2;
        }
    } else {
        fprintf(stderr, "Cannot map states of an uninitialized simulation.\n");
        exit(1);
    }
}

/// Unmap grid state buffers, after having been previously mapped.
void simulation_gpu_states_unmap(simulation_t* simulation) {
    if (simulation->gpu_states.type == STATES_TYPE_OPENGL) {
        gpu_states_opengl_t* opengl = &simulation->gpu_states.gpu_states.opengl;

        if (opengl->gpu_cuda_grid_states_1_mapped) {
            CHECK_ERROR(cudaGraphicsUnmapResources(1, &opengl->gpu_cuda_grid_states_1, simulation->stream));
            opengl->gpu_cuda_grid_states_1_mapped = false;
        }

        if (opengl->gpu_cuda_grid_states_2_mapped) {
            CHECK_ERROR(cudaGraphicsUnmapResources(1, &opengl->gpu_cuda_grid_states_2, simulation->stream));
            opengl->gpu_cuda_grid_states_2_mapped = false;
        }
    } else if (simulation->gpu_states.type == STATES_TYPE_CUDA) {
        // noop
    } else {
        fprintf(stderr, "Cannot unmap states of an uninitialized simulation.\n");
        exit(1);
    }
}

/// Write a ruleset to a file.
void ruleset_save(char* filename, u8* ruleset) {
    if (!file_write(filename, ruleset, get_ruleset_size())) {
        fprintf(stderr, "Failed to write a ruleset.\n");
        exit(1);
    }
}

/// Load a ruleset from a file to a pre-allocated buffer.
void ruleset_load(char* filename, u8* ruleset) {
    if (!file_load(filename, ruleset, get_ruleset_size())) {
        fprintf(stderr, "Failed to load a ruleset.\n");
        exit(1);
    }
}

/// Assigns the current grid from the provided `cpu_grid` and `gpu_grid`. Useful for resetting simulations to an initial state.
void simulation_set_grid(simulation_t* simulation, u8* cpu_grid, u8* gpu_grid) {
    if (CPU_VERIFY) {
        memcpy(simulation->cpu_grid_states_1, cpu_grid, GRID_AREA_WITH_PITCH * sizeof(u8));
    }

    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    cudaMemcpyAsync(gpu_grid_states_1, gpu_grid, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyDeviceToDevice, simulation->stream);
    simulation_gpu_states_unmap(simulation);
}

/// Copy the simulation's CPU grid to the GPU grid.
void simulation_copy_grid_cpu_gpu(simulation_t* simulation) {
    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    cudaMemcpyAsync(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream);
    simulation_gpu_states_unmap(simulation);
}

/// Copy a ruleset from/to/on the GPU.
void ruleset_copy(u8* to, u8* from, cudaMemcpyKind kind, cudaStream_t stream) {
    CHECK_ERROR(cudaMemcpyAsync(to, from, get_ruleset_size() * sizeof(u8), kind, stream));
}

/// Copy the simulation's ruleset from the GPU to the CPU.
void simulation_copy_ruleset_gpu_cpu(simulation_t* simulation) {
    ruleset_copy(simulation->cpu_ruleset, simulation->gpu_ruleset, cudaMemcpyDeviceToHost, simulation->stream);
}

/// Copy the simulation's ruleset from the CPU to the GPU.
void simulation_copy_ruleset_cpu_gpu(simulation_t* simulation) {
    ruleset_copy(simulation->gpu_ruleset, simulation->cpu_ruleset, cudaMemcpyHostToDevice, simulation->stream);
}

/// Save the ruleset to a file.
void simulation_ruleset_save(simulation_t* simulation, char* filename) {
    simulation_copy_ruleset_gpu_cpu(simulation);
    cudaStreamSynchronize(simulation->stream);
    ruleset_save(filename, simulation->cpu_ruleset);
}

/// Load the ruleset from a file.
void simulation_ruleset_load(simulation_t* simulation, char* filename) {
    ruleset_load(filename, simulation->cpu_ruleset);
    simulation_copy_ruleset_cpu_gpu(simulation);
    cudaStreamSynchronize(simulation->stream);
}

/// Save the grid to a file.
void grid_save(char* filename, u8* grid) {
    if (!file_write_2d_pitch(filename, grid, GRID_WIDTH, GRID_HEIGHT, GRID_PITCH)) {
        fprintf(stderr, "Failed to write a grid.\n");
        exit(1);
    }
}

/// Load the grid from a file.
void grid_load(char* filename, u8* grid) {
    if (!file_load_2d_pitch(filename, grid, GRID_WIDTH, GRID_HEIGHT, GRID_PITCH)) {
        fprintf(stderr, "Failed to load a grid.\n");
        exit(1);
    }
}

/// Save the simulation grid to a file.
void simulation_grid_save(simulation_t* simulation, char* filename) {
    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    CHECK_ERROR(cudaMemcpyAsync(simulation->cpu_grid_states_1, gpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyDeviceToHost, simulation->stream));
    cudaStreamSynchronize(simulation->stream);
    simulation_gpu_states_unmap(simulation);
    grid_save(filename, simulation->cpu_grid_states_1);
}

/// Load the simulation grid from a file.
void simulation_grid_load(simulation_t* simulation, char* filename) {
    u8* gpu_grid_states_1;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, NULL);
    grid_load(filename, simulation->cpu_grid_states_1);
    CHECK_ERROR(cudaMemcpyAsync(gpu_grid_states_1, simulation->cpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice, simulation->stream));
    cudaStreamSynchronize(simulation->stream);
    simulation_gpu_states_unmap(simulation);
}

/// Compute the fitness of the current iteration from the reduced total number of "fit cells" `cpu_fit_cells`.
void simulation_compute_fitness(simulation_t* simulation, u32 iteration) {
    f32 proportion_actual = ((f32) simulation->cpu_fit_cells) / ((f32) GRID_AREA);
    f32 fitness;

    // See https://www.desmos.com/calculator/b8daos2dqt
    f32 x = proportion_actual; // aliases
    f32 p = get_fitness_fn_target(iteration);

    if (FITNESS_FN_TYPE == FITNESS_FN_TYPE_ABS) {
        fitness = 1.0f - abs(x - p);
    } else if (FITNESS_FN_TYPE == FITNESS_FN_TYPE_LINEAR) {
        if (p == 0.0f) {
            fitness = (1.0f - x) / (1.0f - p);
        } else if (p == 1.0f) {
            fitness = x / p;
        } else {
            fitness = min(x / p, (1.0f - x) / (1.0f - p));
        }
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

/// Reset the cumulative error of the simulation.
void simulation_cumulative_error_reset(simulation_t* simulation) {
    simulation->cumulative_error = 0.0f;
}

/// Compute the current error and add it to the cumulative error.
/**
 *  The current error is computed from the fitness \f$f\f$ as: \f$(1 - f)^2\f$,
 *  which is then added to the cumulative error.
 */
void simulation_cumulative_error_add(simulation_t* simulation) {
    f32 error_sqrt = 1.0f - simulation->fitness;
    simulation->cumulative_error += error_sqrt * error_sqrt;
}

/// Normalize the cumulative error by dividing it by the number of iterations in which fitness was computed.
void simulation_cumulative_error_normalize(simulation_t* simulation) {
    simulation->cumulative_error /= (f32) get_fitness_iterations();
}

/// Ensure enough storage size for the `temp_storage_t`.
void simulation_temp_storage_ensure(simulation_t* simulation, size_t size) {
    temp_storage_ensure(&simulation->temp_storage, size, simulation->stream);
}

/// Sum the number of fit cells, on the GPU.
/**
 *  Performs a reduction on the number of fit cells per block `gpu_fit_cells_block_sums` and stores the resulting sum in `gpu_fit_cells`.
 */
void simulation_reduce_fit_cells_async(simulation_t* simulation) {
    // Ensure enough temp storage
    size_t temp_storage_size = 0;

    CHECK_ERROR(cub::DeviceReduce::Sum(NULL, temp_storage_size, simulation->gpu_fit_cells_block_sums, simulation->gpu_fit_cells, GRID_AREA_IN_BLOCKS));
    simulation_temp_storage_ensure(simulation, temp_storage_size);

    // Reduce
    CHECK_ERROR(cub::DeviceReduce::Sum(simulation->temp_storage.allocation, simulation->temp_storage.size, simulation->gpu_fit_cells_block_sums, simulation->gpu_fit_cells, GRID_AREA_IN_BLOCKS, simulation->stream, CUB_DEBUG_SYNCHRONOUS));

    // Debug
    CHECK_ERROR(cudaMemcpyAsync(&simulation->cpu_fit_cells, simulation->gpu_fit_cells, sizeof(u32), cudaMemcpyDeviceToHost, simulation->stream));
}
