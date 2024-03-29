/**
 * @file
 * @brief Entry point of the program and kernels.
 */

#include <algorithm>
#include <boost/math/distributions/fwd.hpp>
#include <signal.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <vector>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <boost/math/distributions/binomial.hpp>
#include "common.h"
#include "config.h"
#include "config_derived.h"
#include "util.cuh"
#include "draw.cuh"
#include "math.cuh"
#include "simulation.cuh"

using namespace std;
using namespace cooperative_groups;

/// A ruleset that is kept around between selections.
/**
  If a ruleset is ranked less than `POPULATION_ELITES` during selection, it is kept
  for the selection phase of the next iteration. This ensures that the best rulesets
  are never forgotten.
   */
typedef struct {
    /// The currently stored ruleset. Undefined in the 0th iteration.
    u8* gpu_ruleset;
    /// The cumulative error of the ruleset currently stored in `gpu_ruleset`. `NaN` in the 0th iteration.
    f32 cumulative_error;
} elite_ruleset_t;

/// Holds the state for the genetic algorithm that searches for the best rulesets that best fit the given fitness function.
typedef struct {
    /* CPU */
    /// CUB temporary storage
    temp_storage_t temp_storage;
    /// A CUDA stream per candidate
    array<cudaStream_t, POPULATION_SIZE> streams;
    /// RNG states
    array<curandStatePhilox4_32_10_t*, CROSSOVER_MUTATE_KERNELS_MAX> cpu_gpu_rngs;
    /// CPU-allocated initial states of the grid.
    vector<u8> cpu_initial_grid;
    /// CPU-allocated POPULATION_SELECTION-long array of pointers to mutation rulesets.
    /// These rulesets are used for:
    /// * copying the top POPULATION_ELITES rulesets into elite rulesets;
    /// * reading from during mutation, while new rulesets are written to simulation rulesets.
    array<u8*, POPULATION_SELECTION> cpu_gpu_mutation_rulesets;
    /// Rulesets which are passed on to the next population without any mutations.
    array<elite_ruleset_t, POPULATION_ELITES> elite_rulesets;
    /// Containers for simulations of rulesets on the GPU. Used to determine the `cumulative_error`,
    /// by which simulations are selected.
    array<simulation_t, POPULATION_SIZE> simulations;
    /// The current index of the population being simulated.
    u32 population_index;

    /* GPU */
    /// GPU-allocated initial states of the grid.
    u8* gpu_initial_grid;
    /// GPU-allocated POPULATION_SIZE_PLUS_ELITES-long array of pointers to simulation and elite rulesets. Used for sorting during selection.
    u8** gpu_gpu_selection_rulesets_unordered;
    /// GPU-allocated POPULATION_SIZE_PLUS_ELITES-long array of pointers to simulation and elite rulesets. Used for sorting during selection.
    u8** gpu_gpu_selection_rulesets_ordered;
    /// GPU-allocated POPULATION_SIZE_PLUS_ELITES-long array of cumulative errors of simulation and elite rulesets. Used for sorting during selection.
    f32* gpu_selection_cumulative_errors_unordered;
    /// GPU-allocated POPULATION_SIZE_PLUS_ELITES-long array of cumulative errors of simulation and elite rulesets. Used for sorting during selection.
    f32* gpu_selection_cumulative_errors_ordered;
    /// GPU-allocated POPULATION_SELECTION-long array of pointers to mutation rulesets.
    u8** gpu_gpu_mutation_rulesets;
    /// GPU-allocated POPULATION_ELITES-long array of pointers to elite rulesets.
    u8** gpu_gpu_elite_rulesets;
} seeker_t;

/*
 * Global state
 */
/// Time measurement events
cudaEvent_t start, stop;
/// Whether an interrupt signal was received (^C)
sig_atomic_t sigint_received = 0;

/// Fills the neighbour array `neighbours` with relative offsets of neighbouring cells, depending on the cell position `x`, `y`.
__inline__ __host__ __device__ void get_neighbours(i32 x, i32 y, i32vec2 neighbours[CELL_NEIGHBOURHOOD_SIZE]) {
#if GRID_GEOMETRY == GRID_GEOMETRY_SQUARE
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
    neighbours[0] = make_i32vec2(-1, -1);
    neighbours[1] = make_i32vec2( 0, -1);
    neighbours[2] = make_i32vec2( 1, -1);
    neighbours[3] = make_i32vec2(-1,  0);
    neighbours[4] = make_i32vec2( 1,  0);
    neighbours[5] = make_i32vec2(-1,  1);
    neighbours[6] = make_i32vec2( 0,  1);
    neighbours[7] = make_i32vec2( 1,  1);
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
    neighbours[0] = make_i32vec2(-1,  0);
    neighbours[1] = make_i32vec2( 0, -1);
    neighbours[2] = make_i32vec2( 1,  0);
    neighbours[3] = make_i32vec2( 0,  1);
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE
    bool pointing_up = (x + y) % 2 == 0;
    #if CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX
    if (pointing_up) {
        neighbours[ 0] = make_i32vec2(-1,  0);
        neighbours[ 1] = make_i32vec2(-1, -1);
        neighbours[ 2] = make_i32vec2( 0, -1);
        neighbours[ 3] = make_i32vec2( 1, -1);
        neighbours[ 4] = make_i32vec2( 1,  0);
        neighbours[ 5] = make_i32vec2( 2,  0);
        neighbours[ 6] = make_i32vec2( 2,  1);
        neighbours[ 7] = make_i32vec2( 1,  1);
        neighbours[ 8] = make_i32vec2( 0,  1);
        neighbours[ 9] = make_i32vec2(-1,  1);
        neighbours[10] = make_i32vec2(-2,  1);
        neighbours[11] = make_i32vec2(-2,  0);
    } else {
        neighbours[ 0] = make_i32vec2(-1,  0);
        neighbours[ 1] = make_i32vec2(-2,  0);
        neighbours[ 2] = make_i32vec2(-2, -1);
        neighbours[ 3] = make_i32vec2(-1, -1);
        neighbours[ 4] = make_i32vec2( 0, -1);
        neighbours[ 5] = make_i32vec2( 1, -1);
        neighbours[ 6] = make_i32vec2( 2, -1);
        neighbours[ 7] = make_i32vec2( 2,  0);
        neighbours[ 8] = make_i32vec2( 1,  0);
        neighbours[ 9] = make_i32vec2( 1,  1);
        neighbours[10] = make_i32vec2( 0,  1);
        neighbours[11] = make_i32vec2(-1,  1);
    }
    #elif CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE
    if (pointing_up) {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2( 1,  0);
        neighbours[2] = make_i32vec2( 0,  1);
    } else {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2( 1,  0);
        neighbours[2] = make_i32vec2( 0, -1);
    }
    #endif
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    bool row_even = y % 2 == 0;

    if (row_even) {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2(-1, -1);
        neighbours[2] = make_i32vec2( 0, -1);
        neighbours[3] = make_i32vec2( 1,  0);
        neighbours[4] = make_i32vec2( 0,  1);
        neighbours[5] = make_i32vec2(-1,  1);
    } else {
        neighbours[0] = make_i32vec2(-1,  0);
        neighbours[1] = make_i32vec2( 0, -1);
        neighbours[2] = make_i32vec2( 1, -1);
        neighbours[3] = make_i32vec2( 1,  0);
        neighbours[4] = make_i32vec2( 1,  1);
        neighbours[5] = make_i32vec2( 0,  1);
    }
#endif
}

/// Prints some primary settings of the configuration `config.h`.
void print_configuration() {
    if (POPULATION_SELECTION > POPULATION_SIZE) {
        printf("`POPULATION_SELECTION` (%d) may not exceed `POPULATION_SIZE` (%d).", POPULATION_SELECTION, POPULATION_SIZE);
    }

    if (POPULATION_SELECTION < 2) {
        printf("`POPULATION_SELECTION` (%d) must be set to at least 2.", POPULATION_SELECTION);
    }

    if (POPULATION_ELITES > POPULATION_SELECTION) {
        printf("`POPULATION_ELITES` (%d) may not exceed `POPULATION_SELECTION` (%d).", POPULATION_ELITES, POPULATION_SELECTION);
    }

    printf("\nConfiguration:\n");
    printf("\tGrid width: %d\n", GRID_WIDTH);
    printf("\tGrid height: %d\n", GRID_HEIGHT);

    if (GRID_GEOMETRY == GRID_GEOMETRY_SQUARE) {
        printf("\tGrid geometry: Square\n");
    } else if (GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE) {
        printf("\tGrid geometry: Triangle\n");
    } else if (GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON) {
        printf("\tGrid geometry: Hexagon\n");
    } else {
        printf("\tGrid geometry: Invalid, aborting...\n");
        exit(1);
    }

    if (CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_VERTEX) {
        printf("\tCell neighbourhood type: Vertex\n");
    } else if (CELL_NEIGHBOURHOOD_TYPE == CELL_NEIGHBOURHOOD_TYPE_EDGE) {
        printf("\tCell neighbourhood type: Edge\n");
    } else {
        printf("\tCell neighbourhood type: Invalid, aborting...\n");
        exit(1);
    }

    printf("\tCell neighbourhood size: %d\n", CELL_NEIGHBOURHOOD_SIZE);
    printf("\tCell states: %d\n", CELL_STATES);

    CPU_CELL_NEIGHBOURHOOD_COMBINATIONS = compute_neighbouring_state_combinations(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES);
    cudaMemcpyToSymbol(GPU_CELL_NEIGHBOURHOOD_COMBINATIONS, &CPU_CELL_NEIGHBOURHOOD_COMBINATIONS, sizeof(int));
    printf("\tCell neighbourhood combinations: %d (with a combinatorial number system, %ld with simple indexing)\n", CPU_CELL_NEIGHBOURHOOD_COMBINATIONS, powli(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES));

    CPU_RULESET_SIZE = compute_ruleset_size(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES);
    cudaMemcpyToSymbol(GPU_RULESET_SIZE, &CPU_RULESET_SIZE, sizeof(int));
    printf("\tRuleset size: %d (with a combinatorial number system, %ld with simple indexing)\n", CPU_RULESET_SIZE, CELL_STATES * powli(CELL_NEIGHBOURHOOD_SIZE, CELL_STATES));
    printf("\tshared_subgrid_margin: %d\n", SHARED_SUBGRID_MARGIN);
    printf("\tshared_subgrid_length: %d\n", SHARED_SUBGRID_LENGTH);
    printf("\tshared_subgrid_area: %d\n", SHARED_SUBGRID_AREA);
    printf("\tshared_subgrid_load_iterations: %d\n", SHARED_SUBGRID_LOAD_ITERATIONS);
    printf("\n");
    printf("CPU verification: %s\n", CPU_VERIFY ? "ENABLED -- expect performance impact" : "disabled");
    printf("\n");
}

/// Calculates the 1D index of a cell within shared memory.
__inline__ __host__ __device__ i32 get_cell_index_shared(i32 x, i32 y) {
    assert(x >= 0);
    assert(x < SHARED_SUBGRID_LENGTH);
    assert(y >= 0);
    assert(y < SHARED_SUBGRID_LENGTH);

    return x + y * SHARED_SUBGRID_LENGTH;
}

/// Returns a 1D index into a 2D row-aligned (pitched) grid.
__inline__ __host__ __device__ i32 get_cell_index(i32 x, i32 y) {
    x = mod(x, GRID_WIDTH);
    y = mod(y, GRID_HEIGHT);

    return x + y * GRID_PITCH;
}

/// Returns whether the cell is considered "fit" based on its state and/or state change.
__inline__ __host__ __device__ bool cell_state_fit(u8 state_prev, u8 state_next) {
    if (FITNESS_EVAL == FITNESS_EVAL_STATE) {
        return state_next == FITNESS_EVAL_STATE_INDEX;
    } else if (FITNESS_EVAL == FITNESS_EVAL_UPDATE) {
        return state_prev != state_next;
    } else {
        return false;
    }
}

/// Computes the next state for the given cell, using the provided ruleset.
__host__ __device__ u8 get_next_state(u8 current_state, u8* neighbours, u8* ruleset) {
    // In debug mode, validate the `current_state` argument.
    assert(current_state < CELL_STATES);

    // In debug mode, validate the `neighbours` argument.
#ifndef NDEBUG
    {
        u8 total_neighbours = 0;

        for (u8 state = 0; state < CELL_STATES; state++) {
            u8 current_neighbours = neighbours[state];
            total_neighbours += current_neighbours;

            assert(current_neighbours <= CELL_NEIGHBOURHOOD_SIZE);
        }

        assert(total_neighbours == CELL_NEIGHBOURHOOD_SIZE);
    }
#endif

    u32 index = get_rule_index(get_cell_neighbourhood_combinations(), current_state, CELL_STATES, neighbours, CELL_NEIGHBOURHOOD_SIZE);

    assert(index < get_ruleset_size());

    return ruleset[index];
}

/// Updates the given cell using the provided ruleset, while calculating whether the cell is considered "fit". Does **not** use shared memory.
__host__ __device__ bool update_cell(u8* in_grid, u8* out_grid, u8* ruleset, i32 x, i32 y) {
    i32 cell_index = get_cell_index(x, y);
    u8 current_state = in_grid[cell_index];
    i32vec2 neighbours[CELL_NEIGHBOURHOOD_SIZE];
    u8 state_count[CELL_STATES] = { 0 };

    get_neighbours(x, y, neighbours);

    for (u32 neighbour_index = 0; neighbour_index < CELL_NEIGHBOURHOOD_SIZE; neighbour_index++) {
        i32vec2 neighbour = neighbours[neighbour_index];
        i32 abs_x = x + neighbour.x;
        i32 abs_y = y + neighbour.y;
        i32 neighbour_cell_index = get_cell_index(abs_x, abs_y);

        state_count[in_grid[neighbour_cell_index]] += 1;
    }

    u8 next_state = get_next_state(current_state, state_count, ruleset);
    out_grid[cell_index] = next_state;

    return cell_state_fit(current_state, next_state);
}

/// Updates the given cell using the provided ruleset, while calculating whether the cell is considered "fit". Does use shared memory.
__device__ bool update_cell_shared(u8* in_subgrid_shared, u8* out_grid, u8* ruleset, i32 x_global, i32 y_global, i32 x_shared, i32 y_shared) {
    i32 cell_index_shared = get_cell_index_shared(x_shared, y_shared);
    i32 cell_index_global = get_cell_index(x_global, y_global);
    u8 current_state = in_subgrid_shared[cell_index_shared];
    i32vec2 neighbours[CELL_NEIGHBOURHOOD_SIZE];
    u8 state_count[CELL_STATES] = { 0 };

    get_neighbours(x_global, y_global, neighbours);

    for (u32 neighbour_index = 0; neighbour_index < CELL_NEIGHBOURHOOD_SIZE; neighbour_index++) {
        i32vec2 neighbour_offset = neighbours[neighbour_index];
        i32 abs_x_shared = x_shared + neighbour_offset.x;
        i32 abs_y_shared = y_shared + neighbour_offset.y;
        i32 neighbour_cell_index_shared = get_cell_index_shared(abs_x_shared, abs_y_shared);
        u8 neighbour_state = in_subgrid_shared[neighbour_cell_index_shared];
        state_count[neighbour_state] += 1;
    }

    u8 next_state = get_next_state(current_state, state_count, ruleset);
    out_grid[cell_index_global] = next_state;

    return cell_state_fit(current_state, next_state);
}

/// Simulates a single iteration on the CPU.
/**
 * Simulates a single iteration of the cellular automaton according to the provided ruleset.
 * Performs partial collection of fit cells according to the fit cell criterion `FITNESS_EVAL`.
 *
 *   in_grid - input grid state
 *   out_grid - output grid state
 *   ruleset - the ruleset to use for computing the `out_grid`
 *   fit_cells_block_sum - a pointer to a counter of the total fit cells, or `NULL` if none should be collected
 */
__host__ void cpu_simulate_step(u8* in_grid, u8* out_grid, u8* ruleset, u32* fit_cells_block_sum) {
    if (fit_cells_block_sum) {
        *fit_cells_block_sum = 0;
    }

    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            bool fit = update_cell(in_grid, out_grid, ruleset, x, y);

            if (fit_cells_block_sum) {
                *fit_cells_block_sum += (u32) fit;
            }
        }
    }
}

/// Simulates a single iteration on the GPU using shared memory.
/*
 * Simulates a single iteration of the cellular automaton according to the provided ruleset.
 * Performs partial collection of fit cells according to the fit cell criterion `FITNESS_EVAL`.
 *
 *   in_grid - input grid state
 *   out_grid - output grid state
 *   ruleset - the ruleset to use for computing the `out_grid`
 *   fit_cells_block_sum - an array of fit cells per block to, each block writes to the corresponding index, or `NULL` if none should be collected
 */
__global__ void gpu_simulate_step_kernel_shared(u8* in_grid, u8* out_grid, u8* ruleset, u32* fit_cells_block_sums) {
    __shared__ u8 shared_data[SHARED_SUBGRID_AREA];

    // load shared_subgrid
    i32 x0 = blockIdx.x * BLOCK_LENGTH - SHARED_SUBGRID_MARGIN;
    i32 y0 = blockIdx.y * BLOCK_LENGTH - SHARED_SUBGRID_MARGIN;

    for (i32 i = 0; i < SHARED_SUBGRID_LOAD_ITERATIONS; i++) {
        i32 index_shared = threadIdx.x + threadIdx.y * BLOCK_LENGTH + i * BLOCK_AREA;

        if (index_shared < SHARED_SUBGRID_AREA) {
            i32 x = x0 + index_shared % SHARED_SUBGRID_LENGTH;
            i32 y = y0 + index_shared / SHARED_SUBGRID_LENGTH;
            i32 index_global = get_cell_index(x, y);

            shared_data[index_shared] = in_grid[index_global];
        }
    }

    __syncthreads(); // ensure input subgrid is loaded into shared memory before continuing

    i32 x = threadIdx.x + blockIdx.x * BLOCK_LENGTH;
    i32 y = threadIdx.y + blockIdx.y * BLOCK_LENGTH;
    i32 x_shared = threadIdx.x + SHARED_SUBGRID_MARGIN;
    i32 y_shared = threadIdx.y + SHARED_SUBGRID_MARGIN;
    bool fit = false;

    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        fit = update_cell_shared(shared_data, out_grid, ruleset, x, y, x_shared, y_shared);
    }

    if (fit_cells_block_sums == NULL) {
        return;
    }

    // initialize fit_cells_block_sum to 0
    i32 block_index_1d = blockIdx.x + blockIdx.y * gridDim.x;
    fit_cells_block_sums[block_index_1d] = 0;

    /*
    Until now, `shared_data` was used for input states.
    Past this `__syncthreads` call, it is used for a parallel reduction of fit states.
     */
    __syncthreads();

    // interpret `shared_data` as `u8*`, `u16*` and `u32*` based on the iteration of the parallel reduction
    i32 fit_index = threadIdx.x + threadIdx.y * BLOCK_LENGTH;
    u8* shared_fit_cells_u8 = shared_data;
    u16* shared_fit_cells_u16 = (u16*) shared_data;
    u32* shared_fit_cells_u32 = (u32*) shared_data;

    shared_fit_cells_u8[fit_index] = (u8) fit;

    __syncthreads();

    i32 step = BLOCK_AREA / 2;
    i32 i = 0;

    #pragma unroll
    while (step != 0) {
        bool in_bounds = fit_index < step;

        if (i == 0) {
            u16 sum;

            if (in_bounds) {
                sum = shared_fit_cells_u8[fit_index] + shared_fit_cells_u8[fit_index + step];
            }

            __syncthreads();

            if (in_bounds) {
                shared_fit_cells_u16[fit_index] = sum;
            }
        } else if (i == 1) {
            u32 sum;

            if (in_bounds) {
                sum = shared_fit_cells_u16[fit_index] + shared_fit_cells_u16[fit_index + step];
            }

            __syncthreads();

            if (in_bounds) {
                shared_fit_cells_u32[fit_index] = sum;
            }
        } else {
            if (in_bounds) {
                shared_fit_cells_u32[fit_index] += shared_fit_cells_u32[fit_index + step];
            }
        }

        __syncthreads(); // synchronizace vláken po provedení každé fáze

        step /= 2;
        i++;
    }

    // No need to synchronize, that is done after each iteration of the preceding loop.
    fit_cells_block_sums[block_index_1d] = shared_fit_cells_u32[0];
}

/// Simulates a single iteration on the GPU using **no** shared memory.
__global__ void gpu_simulate_step_kernel_noshared(u8* in_grid, u8* out_grid, u8* ruleset, u32* fit_cells_block_sums) {
    i32 x = threadIdx.x + blockIdx.x * BLOCK_LENGTH;
    i32 y = threadIdx.y + blockIdx.y * BLOCK_LENGTH;
    bool fit = false;

    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        fit = update_cell(in_grid, out_grid, ruleset, x, y);
    }

    i32 block_index_1d = blockIdx.x + blockIdx.y * gridDim.x;

    if (fit_cells_block_sums == NULL) {
        return;
    }

    // initialize fit_cells_block_sum to 0
    fit_cells_block_sums[block_index_1d] = 0;

    __syncthreads();

    atomicAdd(&fit_cells_block_sums[block_index_1d], (u32) fit);
}

/// Simulates a single iteration of the cellular automaton.
/**
  @param simulation: The simulation which to compute the next iteration for.
  @param async: `false`, if explicit synchronisation should be performed; `true` if multiple simulations are expected to run concurrently.
  @param reduce_fit_cells: Whether the fit cell count should be computed for this iteration.
  */
void simulate_step(simulation_t* simulation, bool async, bool reduce_fit_cells) {
    const i32 STEPS = 1;
    // might as well measure the time since we have to wait for the result anyway
    async &= !CPU_VERIFY;
    reduce_fit_cells |= CPU_VERIFY;
    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, &gpu_grid_states_2);

    if (!async) {
        // Store the initial time.
        CHECK_ERROR(cudaEventRecord(start, simulation->stream));
    }

    // aktualizace simulace + vygenerovani bitmapy pro zobrazeni stavu simulace
    for (i32 i = 0; i < STEPS; i++) {
        if (USE_SHARED_MEMORY) {
            gpu_simulate_step_kernel_shared<<<DIM_BLOCKS, DIM_THREADS, 0, simulation->stream>>>(gpu_grid_states_1, gpu_grid_states_2, simulation->gpu_ruleset, reduce_fit_cells ? simulation->gpu_fit_cells_block_sums : NULL);
        } else {
            gpu_simulate_step_kernel_noshared<<<DIM_BLOCKS, DIM_THREADS, 0, simulation->stream>>>(gpu_grid_states_1, gpu_grid_states_2, simulation->gpu_ruleset, reduce_fit_cells ? simulation->gpu_fit_cells_block_sums : NULL);
        }

        simulation_swap_buffers_gpu(simulation);
        swap(gpu_grid_states_1, gpu_grid_states_2);
    }

    if (!async) {
        // Store the finish time. 
        CHECK_ERROR(cudaEventRecord(stop, simulation->stream));
        CHECK_ERROR(cudaEventSynchronize(stop));

        float elapsedTime;

        // Compute the total update time.
        CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

        if (SHOW_PRINT_UPDATE_TIME) {
            printf("Update: %f ms\n", elapsedTime);
        }
    }

#if CPU_VERIFY
    /* printf("Verifying on the CPU...\n"); */

    u32 fit_cells_expected;

    for (i32 i = 0; i < STEPS; i++) {
        // krok simulace life game na CPU
        cpu_simulate_step(simulation->cpu_grid_states_1, simulation->cpu_grid_states_2, simulation->cpu_ruleset, &fit_cells_expected);
        simulation_swap_buffers_cpu(simulation);
    }

    cudaStreamSynchronize(simulation->stream);
    cudaMemcpyAsync(simulation->cpu_grid_states_tmp, gpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyDeviceToHost, simulation->stream);

    simulation_reduce_fit_cells_async(simulation);
    cudaStreamSynchronize(simulation->stream);

    if (fit_cells_expected != simulation->cpu_fit_cells) {
        fprintf(stderr, "Validation error: difference in sum of fit cells -- %u on GPU, %u on CPU.\n", simulation->cpu_fit_cells, fit_cells_expected);
    }

    // Compare states
    u32 diffs = 0;

    for (i32 y = 0; y < GRID_HEIGHT; y++) {
        for (i32 x = 0; x < GRID_WIDTH; x++) {
            i32 cell_index = get_cell_index(x, y);

            if (simulation->cpu_grid_states_1[cell_index] != simulation->cpu_grid_states_tmp[cell_index]) {
                diffs++;
            }
        }
    }

    if (diffs != 0) {
        fprintf(stderr, "Validation error: %u differences between the GPU and CPU simulation grid.\n", diffs);
    }
#endif

    simulation_gpu_states_unmap(simulation);
}

/// Called every frame
void idle_func() {
    if (!editing_mode) {
        simulate_step(&preview_simulation, false, false);
    }
}

void finalize(void);

void window_close_callback(GLFWwindow* window) {
    finalize();
}

/// Initialize the program
void initialize(int argc, char **argv) {
    char* grid_file = argv[2];

    if (access(grid_file, F_OK) != 0) {
        if (editing_mode) {
            // File may not exist, in that case, we are creating a new grid.
            grid_file = NULL;
        } else {
            fprintf(stderr, "Specified grid file not found: %s\n", grid_file);
            exit(1);
        }
    }

    if (preview_simulation.gpu_states.type != STATES_TYPE_UNDEF) {
        simulation_init(&preview_simulation, true, true, grid_file, randomize_grid, 0);

        if (!editing_mode) {
            if (argc >= 4) {
                printf("Loading ruleset: %s\n", argv[3]);
                simulation_ruleset_load(&preview_simulation, argv[3]);
            } else {
                printf("No path to ruleset provided, using a random ruleset.\n");
            }
        }

        printf("\n");
    }

    // vytvoreni struktur udalosti pro mereni casu
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
}

/// Performed before closing the program
void finalize(void) {
    simulation_free(&preview_simulation);

    // zruseni struktur udalosti
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    finalize_draw();
}

/// SIGINT handler that exits the application immediately
void sigint_handler_abort(int signal) {
    exit(1);
}

/// SIGINT handler that requests the application to close, and if received again, closes the application immediately.
void sigint_handler_soft(int signal) {
    sigint_received += 1;

    if (sigint_received > 1) {
        printf(" Multiple interrupt signals received, aborting immediately.\n");
        exit(1);
    }
}

/// Acknowledge SIGINT received to the user.
void check_sigint(bool* sigint_acknowledged) {
    if (sigint_received > 0 && !*sigint_acknowledged) {
        *sigint_acknowledged = true;
        printf(" Interrupt signal received, finishing current population. Send another signal to abort immediately.\n");
    }
}

/**
 * Simulate the current population to compute the cumulative errors for every ruleset.
 */
void population_simulate(array<simulation_t, POPULATION_SIZE>& simulations) {
    bool sigint_acknowledged = false;

    for (simulation_t& simulation : simulations) {
        simulation_cumulative_error_reset(&simulation);
    }

    u32 fitness_eval_to = get_fitness_eval_to();

    for (u32 iteration = 0; iteration < fitness_eval_to; iteration++) {
        bool reduce_fit_cells = get_fitness_fn_reduce(iteration);

        for (simulation_t& simulation : simulations) {
            simulate_step(&simulation, true, reduce_fit_cells);

            if (reduce_fit_cells) {
                simulation_reduce_fit_cells_async(&simulation);
            }

            check_sigint(&sigint_acknowledged);
        }

        if (reduce_fit_cells) {
            // Wait for the iteration to finish
            for (simulation_t& simulation : simulations) {
                check_sigint(&sigint_acknowledged);

                CHECK_ERROR(cudaStreamSynchronize(simulation.stream));
            }

            // Cumulate error
            for (simulation_t& simulation : simulations) {
                simulation_compute_fitness(&simulation, iteration);
                simulation_cumulative_error_add(&simulation);
            }

            /* for (u32 i = 0; i < POPULATION_SIZE; i++) { */
            /*     simulation_t* simulation = &simulations[i]; */
            /*     printf("simulation[%d] { fit_cells: %d, fitness: %f, cumulative_error: %f }\n", */
            /*             i, simulation->cpu_fit_cells, simulation->fitness, simulation->cumulative_error); */
            /* } */
        }

        check_sigint(&sigint_acknowledged);
    }

    for (simulation_t& simulation : simulations) {
        simulation_cumulative_error_normalize(&simulation);
    }
}

/// Order population by cumulative error.
void seeker_population_order(seeker_t* seeker) {
    bool use_elites = seeker->population_index > 0;

    for (u32 i = 0; i < POPULATION_SIZE; i++) {
        simulation_t* simulation = &seeker->simulations[i];
        u32 selection_index = i;
        u8** gpu_selection_ruleset = &seeker->gpu_gpu_selection_rulesets_unordered[selection_index];
        f32* gpu_selection_cumulative_error = &seeker->gpu_selection_cumulative_errors_unordered[selection_index];

        CHECK_ERROR(cudaMemcpyAsync(gpu_selection_ruleset, &simulation->gpu_ruleset, sizeof(u8*), cudaMemcpyHostToDevice, simulation->stream));
        CHECK_ERROR(cudaMemcpyAsync(gpu_selection_cumulative_error, &simulation->cumulative_error, sizeof(f32), cudaMemcpyHostToDevice, simulation->stream));
    }

    if (use_elites) {
        for (u32 i = 0; i < POPULATION_ELITES; i++) {
            elite_ruleset_t* elite_ruleset = &seeker->elite_rulesets[i];
            u32 selection_index = i + POPULATION_SIZE;
            u8** gpu_selection_ruleset = &seeker->gpu_gpu_selection_rulesets_unordered[selection_index];
            f32* gpu_selection_cumulative_error = &seeker->gpu_selection_cumulative_errors_unordered[selection_index];

            CHECK_ERROR(cudaMemcpyAsync(gpu_selection_ruleset, &elite_ruleset->gpu_ruleset, sizeof(u8*), cudaMemcpyHostToDevice, 0));
            CHECK_ERROR(cudaMemcpyAsync(gpu_selection_cumulative_error, &elite_ruleset->cumulative_error, sizeof(f32), cudaMemcpyHostToDevice, 0));
        }

        /* u8** gpu_selection_rulesets = &seeker->gpu_gpu_selection_rulesets_unordered[POPULATION_SIZE]; */
        /* f32* gpu_selection_cumulative_errors = &seeker->gpu_selection_cumulative_errors_unordered[POPULATION_SIZE]; */
        /* u8** gpu_elite_rulesets = seeker->gpu_gpu_selection_rulesets_ordered; */
        /* f32* gpu_elite_cumulative_errors = seeker->gpu_selection_cumulative_errors_ordered; */

        /* cudaMemcpy(gpu_selection_rulesets, gpu_elite_rulesets, POPULATION_ELITES * sizeof(u8*), cudaMemcpyDeviceToDevice); */
        /* cudaMemcpy(gpu_selection_cumulative_errors, gpu_elite_cumulative_errors, POPULATION_ELITES * sizeof(f32), cudaMemcpyDeviceToDevice); */
    }

    cudaDeviceSynchronize();

    // Order rulesets by cumulative_error
    u32 item_count = use_elites ? POPULATION_SIZE_PLUS_ELITES : POPULATION_SIZE;
    u32 begin_bit = 1; // skip sign bit, error is always unsigned
    u32 end_bit = sizeof(f32) * 8;

    // Get temp storage size
    size_t temp_storage_size = 0;

    CHECK_ERROR(cub::DeviceRadixSort::SortPairs(NULL, temp_storage_size, seeker->gpu_selection_cumulative_errors_unordered, seeker->gpu_selection_cumulative_errors_ordered, seeker->gpu_gpu_selection_rulesets_unordered, seeker->gpu_gpu_selection_rulesets_ordered, item_count, begin_bit, end_bit, 0, CUB_DEBUG_SYNCHRONOUS));
    temp_storage_ensure(&seeker->temp_storage, temp_storage_size, 0);

    // Sort
    CHECK_ERROR(cub::DeviceRadixSort::SortPairs(seeker->temp_storage.allocation, seeker->temp_storage.size, seeker->gpu_selection_cumulative_errors_unordered, seeker->gpu_selection_cumulative_errors_ordered, seeker->gpu_gpu_selection_rulesets_unordered, seeker->gpu_gpu_selection_rulesets_ordered, item_count, begin_bit, end_bit, 0, CUB_DEBUG_SYNCHRONOUS));

    cudaDeviceSynchronize();

    u8* cpu_gpu_selection_rulesets[item_count];
    f32 cpu_selection_cumulative_errors_ordered[item_count];

    CHECK_ERROR(cudaMemcpy(&cpu_gpu_selection_rulesets, seeker->gpu_gpu_selection_rulesets_ordered, item_count * sizeof(u8*), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(&cpu_selection_cumulative_errors_ordered, seeker->gpu_selection_cumulative_errors_ordered, item_count * sizeof(f32), cudaMemcpyDeviceToHost));

    // Ensure data is retrieved from the GPU.
    cudaDeviceSynchronize();

    // Copy rulesets from simulations to mutation ruleset allocations.
    // This effectively performs the selection of POPULATION_SELECTION best candidates.
    for (u32 i = 0; i < POPULATION_SELECTION; i++) {
        u8* gpu_selection_ruleset = cpu_gpu_selection_rulesets[i];
        u8* gpu_mutation_ruleset = seeker->cpu_gpu_mutation_rulesets[i];

        ruleset_copy(gpu_mutation_ruleset, gpu_selection_ruleset, cudaMemcpyDeviceToDevice, 0);
    }

    // Store elite rulesets for the next selection phase.
    for (u32 i = 0; i < POPULATION_ELITES; i++) {
        elite_ruleset_t* elite_ruleset = &seeker->elite_rulesets[i];
        u8* gpu_selection_ruleset = cpu_gpu_selection_rulesets[i];
        u8* gpu_elite_ruleset = elite_ruleset->gpu_ruleset;
        elite_ruleset->cumulative_error = cpu_selection_cumulative_errors_ordered[i];

        ruleset_copy(gpu_elite_ruleset, gpu_selection_ruleset, cudaMemcpyDeviceToDevice, 0);
    }

    // Print errors
    printf("Current errors: ");

    for (u32 i = 0; i < item_count; i++) {
        if (i > 0) {
            printf(", ");
        }

        printf("%f", cpu_selection_cumulative_errors_ordered[i]);
    }

    printf("\n");

    // Ensure all rulesets are copied.
    cudaDeviceSynchronize();
}

/// Initialize PRNGs using the provided seed.
/**
 *  @param rngs: The PRNG states to initialize.
 *  @param seed: The seed to initialize them with.
 */
__global__ void kernel_init_rngs(curandStatePhilox4_32_10_t* rngs, u64 seed) {
    int rng_index = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, rng_index, 0, &rngs[rng_index]);
}

/// Perform crossover using the uniform method `CROSSOVER_METHOD_UNIFORM`.
/**
 *  This kernel generates a random bit for each rule in the rulesets, which is used to decide whether the resulting
 *  rule should be copied from the first or the second ruleset.
 *  Quite expensive.
 *
 *  @param rngs: The PRNGs to use for the generation of random bits.
 *  @param source_ruleset_a: The first source ruleset.
 *  @param source_ruleset_b: The second source ruleset.
 *  @param target_ruleset: The output ruleset the result is written to.
 *  @param item_blocks: Number of items each thread shall process.
 *  @param ruleset_size: The size of the rulesets.
 */
__global__ void kernel_crossover(curandStatePhilox4_32_10_t* rngs, u8* source_ruleset_a, u8* source_ruleset_b, u8* target_ruleset, u32 item_blocks, u32 ruleset_size) {
    int rng_index = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t rng = rngs[rng_index]; // load RNG state from global memory
    u32 rule_index = threadIdx.x + blockIdx.x * CROSSOVER_ITEMS_PER_BLOCK;

    for (u32 item_block = 0; item_block < item_blocks; item_block++) {
        u32vec4 bits128 = curand4(&rng);
        u32 bits32x4[4] = { bits128.x, bits128.y, bits128.z, bits128.w };

        #pragma unroll
        for (u8 e = 0; e < 4; e++) {
            u32 bits32 = bits32x4[e];

            for (u8 i = 0; i < 32; i++) {
                if (rule_index >= ruleset_size) {
                    goto break_outer_loop;
                }

                bool use_a = bits32 & 1;
                u8 rule;

                if (use_a) {
                    rule = source_ruleset_a[rule_index];
                } else {
                    rule = source_ruleset_b[rule_index];
                }

                target_ruleset[rule_index] = rule;

                bits32 >>= 1;
                rule_index += blockDim.x;
            }
        }
    }

break_outer_loop:

    rngs[rng_index] = rng; // store RNG state back to global memory for subsequent kernel calls
}

/// Perform initialization of a ruleset with random rules.
/**
 *  This kernel generates random rules for the provided ruleset.
 *
 *  @param rngs: The PRNGs to use for the generation of random bits.
 *  @param target_ruleset: The output ruleset the result is written to.
 *  @param item_blocks: Number of items each thread shall process.
 *  @param ruleset_size: The size of the rulesets.
 */
__global__ void kernel_randomize_ruleset(curandStatePhilox4_32_10_t* rngs, u8* target_ruleset, u32 item_blocks, u32 ruleset_size) {
    int rng_index = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t rng = rngs[rng_index]; // load RNG state from global memory
    u32 rule_index = threadIdx.x + blockIdx.x * MUTATE_ITEMS_PER_BLOCK;

    for (u32 item_block = 0; item_block < item_blocks; item_block++) {
        u32vec4 rolls_opaque = curand4(&rng);
        u32 rolls[4] = { rolls_opaque.x, rolls_opaque.y, rolls_opaque.z, rolls_opaque.w };

        for (u32 i = 0; i < 4; i++) {
            if (rule_index >= ruleset_size) {
                goto break_outer_loop;
            }

            u32 rule = rolls[i] % CELL_STATES; // FIXME: Not uniform
            target_ruleset[rule_index] = rule;

            rule_index += blockDim.x;
        }
    }

break_outer_loop:

    rngs[rng_index] = rng; // store RNG state back to global memory for subsequent kernel calls
}

/// Perform mutation of a ruleset using the `MUTATION_METHOD_UNIFORM` method.
/**
 *  This kernel generates a random float in the range [0, 1) for each rule in the ruleset, to compare with `mutation_chance`,
 *  and mutates the rule if the float is less than `MUTATION_CHANCE`.
 *  Very expensive compared to other mutation methods, but accurate.
 *
 *  @param rngs: The PRNGs to use for the generation of random bits.
 *  @param target_ruleset: The output ruleset the result is written to.
 *  @param item_blocks: Number of items each thread shall process.
 *  @param ruleset_size: The size of the rulesets.
 *  @param mutation_chance: The change for a rule to mutate.
 */
__global__ void kernel_mutate_uniform(curandStatePhilox4_32_10_t* rngs, u8* target_ruleset, u32 item_blocks, u32 ruleset_size, f32 mutation_chance) {
    int rng_index = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t rng = rngs[rng_index]; // load RNG state from global memory
    u32 rule_index = threadIdx.x + blockIdx.x * MUTATE_ITEMS_PER_BLOCK;

    for (u32 item_block = 0; item_block < item_blocks; item_block++) {
        f32vec4 rolls_opaque = curand_uniform4(&rng);
        f32 rolls[4] = { rolls_opaque.x, rolls_opaque.y, rolls_opaque.z, rolls_opaque.w };

        for (u32 i = 0; i < 4; i++) {
            if (rule_index >= ruleset_size) {
                goto break_outer_loop;
            }

            f32 roll = rolls[i];
            bool mutate = roll < mutation_chance;

            if (mutate) {
                u32 random_int = curand(&rng);
                u8 random_rule = random_int % CELL_STATES; // FIXME: Not uniform
                target_ruleset[rule_index] = random_rule;
            }

            rule_index += blockDim.x;
        }
    }

break_outer_loop:

    rngs[rng_index] = rng; // store RNG state back to global memory for subsequent kernel calls
}

/// Perform mutation of a ruleset using the `MUTATION_METHOD_BINOMIAL_KERNEL` method.
/**
 *  This kernel receives a number of mutations, which is a sample of the binomial distribution of the number of mutations,
 *  and for each mutation, generates a random rule index to mutate.
 *  Very efficient, but inaccurate for high `mutation_chance`s, because the same rule index may be generated multiple times.
 *
 *  @param rngs: The PRNGs to use for the generation of random bits.
 *  @param target_ruleset: The output ruleset the result is written to.
 *  @param item_blocks: Number of items each thread shall process.
 *  @param ruleset_size: The size of the rulesets.
 *  @param mutations: The number of mutations, a sample of the binomial distribution of the number of mutations.
 */
__global__ void kernel_mutate_binomial(curandStatePhilox4_32_10_t* rngs, u8* target_ruleset, u32 item_blocks, u32 ruleset_size, u32 mutations) {
    int rng_index = threadIdx.x + blockIdx.x * blockDim.x;
    curandStatePhilox4_32_10_t rng = rngs[rng_index]; // load RNG state from global memory
    u32 mutation_index = threadIdx.x + blockIdx.x * MUTATE_ITEMS_PER_BLOCK;

    for (u32 item_block = 0; item_block < item_blocks; item_block++) {
        u32vec4 rolls_rule_index_opaque = curand4(&rng);
        u32vec4 rolls_rule_opaque = curand4(&rng);
        u32 rolls_rule_index[4] = { rolls_rule_index_opaque.x, rolls_rule_index_opaque.y, rolls_rule_index_opaque.z, rolls_rule_index_opaque.w };
        u32 rolls_rule[4] = { rolls_rule_opaque.x, rolls_rule_opaque.y, rolls_rule_opaque.z, rolls_rule_opaque.w };

        for (u32 i = 0; i < 4; i++) {
            if (mutation_index >= mutations) {
                goto break_outer_loop;
            }

            u32 rule_index = rolls_rule_index[i] % ruleset_size; // FIXME: Not uniform
            u8 rule = rolls_rule[i] % CELL_STATES; // FIXME: Not uniform

            // Note: This is an intentional race condition, as multiple threads from the same block
            // or different blocks may attempt to write into the same byte.
            target_ruleset[rule_index] = rule;

            mutation_index += blockDim.x;
        }
    }

break_outer_loop:

    rngs[rng_index] = rng;
}

/// Initialize the PRNGs with a random seed.
void seed_rng_set(curandStatePhilox4_32_10_t* rng_set, cudaStream_t stream) {
    u64 seed = random_sample_u64();
    kernel_init_rngs<<<CROSSOVER_MUTATE_BLOCKS_MAX, THREADS_PER_BLOCK, 0, stream>>>(rng_set, seed);
}

/// Randomize all rulesets. See `kernel_randomize_ruleset`.
void seeker_randomize_rulesets(seeker_t* seeker) {
    u32 ruleset_size = get_ruleset_size();
    u32 blocks = min<u32>((ruleset_size + MUTATE_ITEMS_PER_BLOCK - 1) / MUTATE_ITEMS_PER_BLOCK, CROSSOVER_MUTATE_BLOCKS_MAX);
    u32 item_blocks = (ruleset_size + blocks - 1) / blocks;

    if (CPU_VERIFY) {
        if (CELL_STATES >= 0x100) {
            fprintf(stderr, "Cannot verify proper randomization of rulesets for `CELL_STATES` larger than 255.\n");
            exit(1);
        }

        for (simulation_t& simulation : seeker->simulations) {
            CHECK_ERROR(cudaMemsetAsync(simulation.gpu_ruleset, 0xFF, ruleset_size, simulation.stream));
        }

        cudaDeviceSynchronize();
    }

    for (u32 simulation_index = 0; simulation_index < POPULATION_SIZE; simulation_index++) {
        cudaStream_t stream = seeker->streams[simulation_index % CROSSOVER_MUTATE_KERNELS_MAX];
        simulation_t* simulation = &seeker->simulations[simulation_index];
        u32 rng_set_index = simulation_index % CROSSOVER_MUTATE_KERNELS_MAX;
        curandStatePhilox4_32_10_t* rng_set = seeker->cpu_gpu_rngs[rng_set_index];

        seed_rng_set(rng_set, stream);
        kernel_randomize_ruleset<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(rng_set, simulation->gpu_ruleset, item_blocks, ruleset_size);
    }

    if (CPU_VERIFY) {
        cudaDeviceSynchronize();

        for (simulation_t& simulation : seeker->simulations) {
            simulation_copy_ruleset_gpu_cpu(&simulation);
        }

        cudaDeviceSynchronize();
        printf("Verifying ruleset randomization...\n");

        for (u32 simulation_index = 0; simulation_index < POPULATION_SIZE; simulation_index++) {
            simulation_t* simulation = &seeker->simulations[simulation_index];

            for (u32 rule_index = 0; rule_index < ruleset_size; rule_index++) {
                u32 rule = simulation->cpu_ruleset[rule_index];

                if (rule >= CELL_STATES) {
                    fprintf(stderr, "Ruleset randomization verification failed:\n");
                    fprintf(stderr, "Rule #%u of simulation #%u is invalid: %u\n", rule_index, simulation_index, rule);
                    exit(1);
                }
            }
        }

        printf("Ruleset randomization verified.\n");
    }

    cudaDeviceSynchronize();
}

/// Perform crossover and mutation.
/**
 *  The best `POPULATION_SELECTION` of rulesets (with lowest cumulative error) are taken.
 *  Then, `POPULATION_SIZE` rulesets are created using the following way:
 *  - Two rulesets are chosen at random from the selection population, and the crossover method is applied to combine them.
 *  - The mutation method is applied to the resulting combination of rulesets.
 */
void seeker_crossover_mutation(seeker_t* seeker) {
    u32 ruleset_size = get_ruleset_size();
    u32 blocks_crossover = min<u32>((ruleset_size + CROSSOVER_ITEMS_PER_BLOCK - 1) / CROSSOVER_ITEMS_PER_BLOCK, CROSSOVER_MUTATE_BLOCKS_MAX);
    u32 item_blocks_crossover = (ruleset_size + blocks_crossover - 1) / blocks_crossover;
    array<f32, POPULATION_SIZE> mutation_chances;

    cudaDeviceSynchronize();

    // Run CROSSOVER_MUTATE_KERNELS_MAX in parallel
    for (u32 simulation_index = 0; simulation_index < POPULATION_SIZE; simulation_index++) {
        cudaStream_t stream = seeker->streams[simulation_index % CROSSOVER_MUTATE_KERNELS_MAX];
        simulation_t* simulation = &seeker->simulations[simulation_index];
        u32 rng_set_index = simulation_index % CROSSOVER_MUTATE_KERNELS_MAX;
        u32 source_ruleset_index_a = random_sample_u32(POPULATION_SELECTION);
        u32 source_ruleset_index_b = random_sample_u32(POPULATION_SELECTION);

        /* while (source_ruleset_index_a == source_ruleset_index_b) { */
        /*     source_ruleset_index_b = random_sample_u32(POPULATION_SELECTION); */
        /* } */

        u8* source_ruleset_a = seeker->cpu_gpu_mutation_rulesets[source_ruleset_index_a];
        u8* source_ruleset_b = seeker->cpu_gpu_mutation_rulesets[source_ruleset_index_b];
        curandStatePhilox4_32_10_t* rng_set = seeker->cpu_gpu_rngs[rng_set_index];

        f32 mutation_chance;

        if (MUTATION_CHANCE_ADAPTIVE_BY_RANK) {
            // A variation on rank-based adaptive mutation: https://arxiv.org/pdf/2104.08842.pdf
            f32 average_rank = ((f32) (source_ruleset_index_a + source_ruleset_index_b)) * 0.5f;
            mutation_chance = MUTATION_CHANCE * ((average_rank + 1.0f) / (f32) POPULATION_SIZE);
        } else {
            mutation_chance = MUTATION_CHANCE;
        }

        mutation_chances[simulation_index] = mutation_chance;

        if (source_ruleset_index_a == source_ruleset_index_b) {
            // Crossover between the same rulesets
            CHECK_ERROR(cudaMemcpyAsync(simulation->gpu_ruleset, source_ruleset_a, ruleset_size * sizeof(u8), cudaMemcpyDeviceToDevice, stream));
        } else {
            if (CROSSOVER_METHOD == CROSSOVER_METHOD_UNIFORM) {
                seed_rng_set(rng_set, stream);
                kernel_crossover<<<blocks_crossover, THREADS_PER_BLOCK, 0, stream>>>(rng_set, source_ruleset_a, source_ruleset_b, simulation->gpu_ruleset, item_blocks_crossover, ruleset_size);
            } else { // Splice method
                bool order = random_sample_u32(2);

                if (order) {
                    swap(source_ruleset_a, source_ruleset_b);
                }

                std::array<u32, CROSSOVER_METHOD> splice_indices;

                for (u32& splice_index : splice_indices) {
                    splice_index = random_sample_u32(ruleset_size + 1);
                }

                std::sort(splice_indices.begin(), splice_indices.end());

                u32 write_offset = 0;

                for (u32 i = 0; i < CROSSOVER_METHOD; i++) {
                    u32 splice_index = splice_indices[i];
                    u32 difference = splice_index - write_offset;

                    if (difference > 0) {
                        CHECK_ERROR(cudaMemcpyAsync(&simulation->gpu_ruleset[write_offset], &source_ruleset_a[write_offset], difference * sizeof(u8), cudaMemcpyDeviceToDevice, stream));
                    }

                    swap(source_ruleset_a, source_ruleset_b);
                    write_offset = splice_index;
                }

                u32 remaining_length = ruleset_size - write_offset;

                if (remaining_length > 0) {
                    CHECK_ERROR(cudaMemcpyAsync(&simulation->gpu_ruleset[write_offset], &source_ruleset_a[write_offset], remaining_length * sizeof(u8), cudaMemcpyDeviceToDevice, stream));
                }
            }
        }
    }

    u32 blocks_mutate_uniform = min<u32>((ruleset_size + MUTATE_ITEMS_PER_BLOCK - 1) / MUTATE_ITEMS_PER_BLOCK, CROSSOVER_MUTATE_BLOCKS_MAX);
    u32 item_blocks_mutate_uniform = (ruleset_size + blocks_mutate_uniform - 1) / blocks_mutate_uniform;

    // Wait for crossover to finish
    cudaDeviceSynchronize();

    if (MUTATION_METHOD == MUTATION_METHOD_UNIFORM) {
        for (u32 simulation_index = 0; simulation_index < POPULATION_SIZE; simulation_index++) {
            cudaStream_t stream = seeker->streams[simulation_index % CROSSOVER_MUTATE_KERNELS_MAX];
            simulation_t* simulation = &seeker->simulations[simulation_index];
            u32 rng_set_index = simulation_index % CROSSOVER_MUTATE_KERNELS_MAX;
            curandStatePhilox4_32_10_t* rng_set = seeker->cpu_gpu_rngs[rng_set_index];
            f32 mutation_chance = mutation_chances[simulation_index];

            seed_rng_set(rng_set, stream);
            kernel_mutate_uniform<<<blocks_mutate_uniform, THREADS_PER_BLOCK, 0, stream>>>(rng_set, simulation->gpu_ruleset, item_blocks_mutate_uniform, ruleset_size, mutation_chance);
        }
    } else if (MUTATION_METHOD == MUTATION_METHOD_BINOMIAL_MEMSET || MUTATION_METHOD == MUTATION_METHOD_BINOMIAL_KERNEL) {
        for (u32 simulation_index = 0; simulation_index < POPULATION_SIZE; simulation_index++) {
            simulation_t* simulation = &seeker->simulations[simulation_index];

            f32 mutation_chance = mutation_chances[simulation_index];
            f64 n = ruleset_size;
            f64 p = mutation_chance;
            boost::math::binomial_distribution<f64> distribution = boost::math::binomial(n, p);

            f64 uniform = random_sample_f64_normalized();
            f64 binomial = boost::math::quantile(distribution, uniform);
            u32 mutations = (u32) round(binomial);

            if (mutations <= 0) {
                continue;
            }

            if (MUTATION_METHOD == MUTATION_METHOD_BINOMIAL_MEMSET) {
                for (u32 mutation = 0; mutation < mutations; mutation++) {
                    u32 rule_index = random_sample_u32(ruleset_size);
                    u8 rule = (u8) random_sample_u32(CELL_STATES);

                    // First, store the rule in the CPU index. This is necessary, because we need to be able
                    // to continue with the following iterations while the rule is being copied to the GPU.
                    // We wouldn't be able to upload it from the stack.
                    u8* gpu_rule = &simulation->gpu_ruleset[rule_index];
                    u8* cpu_rule = &simulation->cpu_ruleset[rule_index];

                    if (CPU_VERIFY) {
                        *cpu_rule = rule;
                    }

                    /* CHECK_ERROR(cudaMemcpyAsync(gpu_rule, cpu_rule, sizeof(u8), cudaMemcpyHostToDevice, simulation->stream)); */
                    CHECK_ERROR(cudaMemsetAsync(gpu_rule, rule, sizeof(u8), simulation->stream));
                }
            } else if (MUTATION_METHOD == MUTATION_METHOD_BINOMIAL_KERNEL) {
                u32 blocks_mutate_binomial = min<u32>((mutations + MUTATE_ITEMS_PER_BLOCK - 1) / MUTATE_ITEMS_PER_BLOCK, CROSSOVER_MUTATE_BLOCKS_MAX);
                u32 item_blocks_mutate_binomial = (ruleset_size + blocks_mutate_binomial - 1) / blocks_mutate_binomial;

                cudaStream_t stream = seeker->streams[simulation_index % CROSSOVER_MUTATE_KERNELS_MAX];
                simulation_t* simulation = &seeker->simulations[simulation_index];
                u32 rng_set_index = simulation_index % CROSSOVER_MUTATE_KERNELS_MAX;
                u64 seed = random_sample_u64();
                curandStatePhilox4_32_10_t* rng_set = seeker->cpu_gpu_rngs[rng_set_index];

                seed_rng_set(rng_set, stream);
                kernel_mutate_binomial<<<blocks_mutate_binomial, THREADS_PER_BLOCK, 0, stream>>>(rng_set, simulation->gpu_ruleset, item_blocks_mutate_binomial, ruleset_size, mutations);
            } else {
                fprintf(stderr, "unreachable\n");
                exit(1);
            }
        }
    } else {
        fprintf(stderr, "unreachable\n");
        exit(1);
    }

    if (CPU_VERIFY) {
        cudaDeviceSynchronize();

        for (simulation_t& simulation : seeker->simulations) {
            simulation_copy_ruleset_gpu_cpu(&simulation);
        }
    }

    cudaDeviceSynchronize();
}

/// Initialize the seeker state.
void seeker_init(int argc, char **argv, seeker_t* seeker) {
    temp_storage_init(&seeker->temp_storage);

    for (cudaStream_t& stream : seeker->streams) {
        CHECK_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    for (u32 i = 0; i < CROSSOVER_MUTATE_KERNELS_MAX; i++) {
        CHECK_ERROR(cudaMalloc(&seeker->cpu_gpu_rngs[i], sizeof(curandStatePhilox4_32_10_t) * CROSSOVER_MUTATE_RNGS_PER_KERNEL));
    }

    seeker->cpu_initial_grid = vector<u8>(GRID_AREA_WITH_PITCH, 0);

    CHECK_ERROR(cudaMalloc(&seeker->gpu_initial_grid, GRID_AREA_WITH_PITCH * sizeof(u8)));
    CHECK_ERROR(cudaMalloc(&seeker->gpu_gpu_mutation_rulesets, POPULATION_SELECTION * sizeof(u8*)));
    CHECK_ERROR(cudaMalloc(&seeker->gpu_gpu_elite_rulesets, POPULATION_ELITES * sizeof(u8*)));
    CHECK_ERROR(cudaMalloc(&seeker->gpu_gpu_selection_rulesets_unordered, POPULATION_SIZE_PLUS_ELITES * sizeof(u8*)));
    CHECK_ERROR(cudaMalloc(&seeker->gpu_gpu_selection_rulesets_ordered, POPULATION_SIZE_PLUS_ELITES * sizeof(u8*)));
    CHECK_ERROR(cudaMalloc(&seeker->gpu_selection_cumulative_errors_unordered, POPULATION_SIZE_PLUS_ELITES * sizeof(f32)));
    CHECK_ERROR(cudaMalloc(&seeker->gpu_selection_cumulative_errors_ordered, POPULATION_SIZE_PLUS_ELITES * sizeof(f32)));

    // Allocate space for mutation rulesets
    for (i32 i = 0; i < POPULATION_SELECTION; i++) {
        u8** gpu_mutation_ruleset = &seeker->cpu_gpu_mutation_rulesets[i];
        CHECK_ERROR(cudaMalloc(gpu_mutation_ruleset, get_ruleset_size() * sizeof(u8)));
        CHECK_ERROR(cudaMemcpy(&seeker->gpu_gpu_mutation_rulesets[i], gpu_mutation_ruleset, sizeof(u8*), cudaMemcpyHostToDevice));
    }

    // Allocate space for elite rulesets
    for (i32 i = 0; i < POPULATION_ELITES; i++) {
        elite_ruleset_t* elite_ruleset = &seeker->elite_rulesets[i];
        elite_ruleset->cumulative_error = 0.0 / 0.0; // NAN
        CHECK_ERROR(cudaMalloc(&elite_ruleset->gpu_ruleset, get_ruleset_size() * sizeof(u8)));
        CHECK_ERROR(cudaMemcpy(&seeker->gpu_gpu_elite_rulesets[i], &elite_ruleset->gpu_ruleset, sizeof(u8*), cudaMemcpyHostToDevice));
    }

    grid_load(argv[2], &seeker->cpu_initial_grid[0]);

    CHECK_ERROR(cudaMemcpy(seeker->gpu_initial_grid, &seeker->cpu_initial_grid[0], GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice));

    for (simulation_t& simulation : seeker->simulations) {
        simulation_init(&simulation, false, false, NULL, false, 0);
    }

    cudaDeviceSynchronize();

    if (global_argc >= 4) {
        printf("Loading ruleset: %s\n", global_argv[3]);

        for (simulation_t& simulation : seeker->simulations) {
            simulation_ruleset_load(&simulation, global_argv[3]);
        }
    } else {
        printf("No path to ruleset provided, using random rulesets.\n");
        seeker_randomize_rulesets(seeker);
    }

    cudaDeviceSynchronize();

    seeker->population_index = 0;

    printf("Initialized.\n");
}

/// Runs the main seeker loop, which performs the iterations of the genetic algorithm.
void seeker_loop(seeker_t* seeker) {
    signal(SIGINT, sigint_handler_soft);

    while (!sigint_received) {
        // Load initial grid
        printf("Resetting grids...\n");
        for (simulation_t& simulation : seeker->simulations) {
            simulation_set_grid(&simulation, &seeker->cpu_initial_grid[0], seeker->gpu_initial_grid);
        }

        cudaDeviceSynchronize();
        printf("Grids reset finished.\n");

        if (seeker->population_index > 0) {
            printf("Performing selection & mutation...\n");
            seeker_crossover_mutation(seeker);
            printf("Selection & mutation finished.\n");

            cudaDeviceSynchronize();
        }

        printf("Simulating population #%u...\n", seeker->population_index);
        population_simulate(seeker->simulations);
        printf("Finished simulating population #%u.\n", seeker->population_index);

        cudaDeviceSynchronize();

        printf("Ordering population...\n");
        seeker_population_order(seeker);
        printf("Population ordered.\n");

        seeker->population_index++;

        #ifdef EXIT_AFTER_POPULATIONS
        if (seeker->population_index >= EXIT_AFTER_POPULATIONS) {
            break;
        }
        #endif
    }

    printf("Simulations finished.\n");
    signal(SIGINT, sigint_handler_abort);
}

/// Performs disk storage of the best rulesets found.
void seeker_output(seeker_t* seeker) {
    u32 rulesets_to_save;

    printf("How many rulesets would you like to save? [1]: ");

    if (scanf("%u", &rulesets_to_save) != 1) {
        rulesets_to_save = 1;
    }

    if (rulesets_to_save > POPULATION_SIZE) {
        rulesets_to_save = POPULATION_SIZE;
    }

    printf("Transferring rulesets from the GPU...\n");

    for (u32 i = 0; i < rulesets_to_save; i++) {
        simulation_t* simulation = &seeker->simulations[i];
        u8* source_ruleset = seeker->cpu_gpu_mutation_rulesets[i];

        ruleset_copy(simulation->cpu_ruleset, source_ruleset, cudaMemcpyDeviceToHost, simulation->stream);
    }

    cudaDeviceSynchronize();

    printf("Saving rulesets...\n");

    for (u32 i = 0; i < rulesets_to_save; i++) {
        simulation_t* simulation = &seeker->simulations[i];
        char filename[1024];

        snprintf(filename, 1024, "ruleset_%04u.rsr", i);
        ruleset_save(filename, simulation->cpu_ruleset);
    }

    printf("%u rulesets saved.\n", rulesets_to_save);
}

/// Destructor
void seeker_finalize(seeker_t* seeker) {
    for (simulation_t& simulation : seeker->simulations) {
        simulation_free(&simulation);
    }
}

/// Main entry point of the `seek` procedure.
int main_seek(int argc, char **argv) {
    initialize(argc, argv);

    seeker_t seeker;

    seeker_init(argc, argv, &seeker);
    seeker_loop(&seeker);

    if (SAVE_FOUND_RULESETS) {
        seeker_output(&seeker);
    }

    seeker_finalize(&seeker);

    return 0;
}

/// Main entry point of the `show` procedure.
int main_show(int argc, char **argv) {
    init_draw(argc, argv, window_close_callback, idle_func, false);
    initialize(argc, argv);

    return ui_loop();
}

/// Main entry point of the `edit` procedure.
int main_edit(int argc, char** argv) {
    init_draw(argc, argv, window_close_callback, idle_func, true);
    initialize(argc, argv);

    return ui_loop();
}

/// Global main entry point.
int main(int argc, char **argv) {
    global_argc = argc;
    global_argv = argv;

    bool edit = argc >= 3 && strcmp(argv[1], "edit") == 0;
    bool seek = argc >= 3 && strcmp(argv[1], "seek") == 0;
    bool show = argc >= 3 && strcmp(argv[1], "show") == 0;

    if (!edit && !seek && !show) {
        printf("Usage:\n");
        printf("%s edit GRID.rsg [-r]          -- a tool to edit initial grid states, use flag `-r` to randomize\n", argv[0]);
        printf("%s seek GRID.rsg [RULESET.rsr] -- performs search for interesting rulesets\n", argv[0]);
        printf("%s show GRID.rsg [RULESET.rsr] -- performs visual simulation of an existing ruleset\n", argv[0]);
        exit(0);
    }

    print_configuration();
    random_init();

    if (PROMPT_TO_START) {
        printf("Press Enter to begin.");
        getchar();
    }

    if (edit) {
        return main_edit(argc, argv);
    }

    if (seek) {
        return main_seek(argc, argv);
    }

    if (show) {
        return main_show(argc, argv);
    }

    return 0;
}
