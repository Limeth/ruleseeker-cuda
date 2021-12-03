#include <algorithm>
#include <signal.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <array>
#include <vector>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <boost/math/distributions/binomial.hpp>
#include "common.h"
#include "util.cuh"
#include "draw.cuh"
#include "math.cuh"
#include "simulation.cuh"

using namespace std;
using namespace cooperative_groups;

typedef struct {
    u8* gpu_ruleset;
    f32 cumulative_error;
} elite_ruleset_t;

typedef struct {
    /* CPU */
    // CUB temporary storage
    temp_storage_t temp_storage;
    // A CUDA stream per candidate
    array<cudaStream_t, POPULATION_SIZE> streams;
    // RNG states
    array<curandStatePhilox4_32_10_t*, CROSSOVER_MUTATE_KERNELS_MAX> cpu_gpu_rngs;
    // CPU-allocated initial states of the grid.
    vector<u8> cpu_initial_grid;
    // CPU-allocated POPULATION_SELECTION-long array of pointers to mutation rulesets.
    // These rulesets are used for:
    // * copying the top POPULATION_ELITES rulesets into elite rulesets;
    // * reading from during mutation, while new rulesets are written to simulation rulesets.
    array<u8*, POPULATION_SELECTION> cpu_gpu_mutation_rulesets;
    // Rulesets which are passed on to the next population without any mutations.
    array<elite_ruleset_t, POPULATION_ELITES> elite_rulesets;
    // Containers for simulations of rulesets on the GPU. Used to determine the `cumulative_error`,
    // by which simulations are selected.
    array<simulation_t, POPULATION_SIZE> simulations;
    // The current index of the population being simulated.
    u32 population_index;

    /* GPU */
    // GPU-allocated initial states of the grid.
    u8* gpu_initial_grid;
    // GPU-allocated POPULATION_SIZE_PLUS_ELITES-long array of pointers to simulation and elite rulesets,
    // used for sorting during selection.
    u8** gpu_gpu_selection_rulesets_unordered;
    u8** gpu_gpu_selection_rulesets_ordered;
    // GPU-allocated POPULATION_SIZE_PLUS_ELITES-long array of cumulative errors of simulation and elite rulesets,
    // used for sorting during selection.
    f32* gpu_selection_cumulative_errors_unordered;
    f32* gpu_selection_cumulative_errors_ordered;
    // GPU-allocated POPULATION_SELECTION-long array of pointers to mutation rulesets.
    u8** gpu_gpu_mutation_rulesets;
    // GPU-allocated POPULATION_ELITES-long array of pointers to elite rulesets.
    u8** gpu_gpu_elite_rulesets;
} seeker_t;

// A set of rules that has been evaluated with a cumulative error
typedef struct {
    // whether this ruleset was simulated in the current population (false),
    // or simulation was skipped because of its low cumulative error in the previous population (true).
    bool elite;
    // depending on `elite`, either an index into the array of elites, or an index into the array of simulations.
    u32 index;
    // the ruleset's cumulative error, by which it will be ordered.
    f32 cumulative_error;
    // the ruleset
    u8* gpu_ruleset;
} evaluated_ruleset_t;

/*
 * Global state
 */
// Time measurement events
cudaEvent_t start, stop;
// Whether an interrupt signal was received (^C)
sig_atomic_t sigint_received = 0;

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
}

__inline__ __host__ __device__ i32 get_cell_index_shared(i32 x, i32 y) {
    assert(x >= 0);
    assert(x < SHARED_SUBGRID_LENGTH);
    assert(y >= 0);
    assert(y < SHARED_SUBGRID_LENGTH);

    return x + y * SHARED_SUBGRID_LENGTH;
}

// returns an index into a 2D row-aligned array
__inline__ __host__ __device__ i32 get_cell_index(i32 x, i32 y) {
    x = mod(x, GRID_WIDTH);
    y = mod(y, GRID_HEIGHT);

    return x + y * GRID_PITCH;
}

__inline__ __host__ __device__ bool cell_state_fit(u8 state_prev, u8 state_next) {
#if FITNESS_EVAL == FITNESS_EVAL_STATE
    return state_next == FITNESS_EVAL_STATE_INDEX;
#elif FITNESS_EVAL == FITNESS_EVAL_UPDATE
    return state_prev != state_next;
#endif
}

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

    u32 index = get_rule_index(get_cell_neighbourhood_combinations(), current_state, CELL_STATES, neighbours);

    assert(index < get_ruleset_size());

    return ruleset[index];
}

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

/*
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

// Simulates a single iteration
void simulate_step(simulation_t* simulation, bool async, bool reduce_fit_cells) {
    const i32 STEPS = 1;
    // might as well measure the time since we have to wait for the result anyway
    async &= !CPU_VERIFY;
    reduce_fit_cells |= CPU_VERIFY;
    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    simulation_gpu_states_map(simulation, &gpu_grid_states_1, &gpu_grid_states_2);

    if (!async) {
        // ulozeni pocatecniho casu
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
        // ulozeni casu ukonceni simulace
        CHECK_ERROR(cudaEventRecord(stop, simulation->stream));
        CHECK_ERROR(cudaEventSynchronize(stop));

        float elapsedTime;

        // vypis casu simulace
        CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        /* printf("Update: %f ms\n", elapsedTime); */
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

    // porovnani vysledku CPU simulace a GPU simulace
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

// called every frame
void idle_func() {
    simulate_step(&preview_simulation, false, false);
    glutPostRedisplay();
}

void finalize(void);

static void handle_keys(unsigned char key, int x, int y) {
    switch (key) {
        case 27:	// ESC
            finalize();
            exit(0);
    }
}

// inicializace CUDA - alokace potrebnych dat a vygenerovani pocatecniho stavu lifu
void initialize(int argc, char **argv) {
    if (preview_simulation.gpu_states.type != STATES_TYPE_UNDEF) {
        simulation_init(&preview_simulation, true, 0);

        if (argc >= 4) {
            printf("Loading ruleset: %s\n", argv[3]);
            simulation_ruleset_load(&preview_simulation, argv[3]);
        } else {
            printf("No path to ruleset provided, using a random ruleset.\n");
        }

        printf("\n");
    }

    // vytvoreni struktur udalosti pro mereni casu
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
}

// funkce volana pri ukonceni aplikace, uvolni vsechy prostredky alokovane v CUDA 
void finalize(void) {
    simulation_free(&preview_simulation);

    // zruseni struktur udalosti
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    finalize_draw();
}

void sigint_handler_abort(int signal) {
    exit(1);
}

void sigint_handler_soft(int signal) {
    sigint_received += 1;

    if (sigint_received > 1) {
        printf(" Multiple interrupt signals received, aborting immediately.\n");
        exit(1);
    }
}

void check_sigint(bool* sigint_acknowledged) {
    if (sigint_received > 0 && !*sigint_acknowledged) {
        *sigint_acknowledged = true;
        printf(" Interrupt signal received, finishing current population. Send another signal to abort immediately.\n");
    }
}

void population_simulate(array<simulation_t, POPULATION_SIZE>& simulations) {
    bool sigint_acknowledged = false;

    for (simulation_t& simulation : simulations) {
        simulation_cumulative_error_reset(&simulation);
    }

    for (u32 iteration = 0; iteration < FITNESS_EVAL_TO; iteration++) {
        bool reduce_fit_cells = iteration >= FITNESS_EVAL_FROM;

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
                simulation_compute_fitness(&simulation);
                simulation_cumulative_error_add(&simulation);
            }

            /* for (simulation_t& simulation : simulations) { */
            /*     printf("%f, ", simulation.fitness); */
            /* } */

            /* printf("\n"); */
        }

        check_sigint(&sigint_acknowledged);
    }
}

bool comparator(evaluated_ruleset_t& a, evaluated_ruleset_t& b) {
    return a.cumulative_error < b.cumulative_error;
}

void seeker_population_order(seeker_t* seeker) {
    /* for (simulation_t& simulation : seeker->simulations) { */
    /*     simulation_copy_ruleset_gpu_cpu(&simulation); */
    /* } */

    /* // Wait for all rulesets to be copied */
    /* for (simulation_t& simulation : seeker->simulations) { */
    /*     cudaStreamSynchronize(simulation.stream); */
    /* } */

    /* std::vector<evaluated_ruleset_t> evaluated_rulesets; */

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

__global__ void kernel_init_rngs(curandStatePhilox4_32_10_t* rngs, u64 seed) {
    curand_init(seed, threadIdx.x, 0, &rngs[blockIdx.x]);
}

__global__ void kernel_crossover(curandStatePhilox4_32_10_t* rngs, u8* source_ruleset_a, u8* source_ruleset_b, u8* target_ruleset, u32 item_blocks, u32 ruleset_size) {
    thread_block block = this_thread_block();
    curandStatePhilox4_32_10_t rng = rngs[blockIdx.x]; // load RNG state from global memory
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

    rngs[blockIdx.x] = rng; // store RNG state back to global memory for subsequent kernel calls
}

__global__ void kernel_mutate(curandStatePhilox4_32_10_t* rngs, u8* target_ruleset, u32 item_blocks, u32 ruleset_size) {
    thread_block block = this_thread_block();
    curandStatePhilox4_32_10_t rng = rngs[blockIdx.x]; // load RNG state from global memory
    u32 rule_index = threadIdx.x + blockIdx.x * MUTATE_ITEMS_PER_BLOCK;

    for (u32 item_block = 0; item_block < item_blocks; item_block++) {
        f32vec4 rolls_opaque = curand_uniform4(&rng);
        f32 rolls[4] = { rolls_opaque.x, rolls_opaque.y, rolls_opaque.z, rolls_opaque.w };

        for (u32 i = 0; i < 4; i++) {
            if (rule_index >= ruleset_size) {
                goto break_outer_loop;
            }

            f32 roll = rolls[i];
            bool mutate = roll < MUTATION_CHANCE;

            if (mutate) {
                u32 random_int = curand(&rng);
                u8 random_rule = random_int % CELL_STATES;
                target_ruleset[rule_index] = random_rule;
            }

            rule_index += blockDim.x;
        }
    }

break_outer_loop:

    rngs[blockIdx.x] = rng; // store RNG state back to global memory for subsequent kernel calls
}

void seeker_crossover_mutation(seeker_t* seeker) {
    u32 ruleset_size = get_ruleset_size();
    u32 blocks_crossover = min<u32>((ruleset_size + CROSSOVER_ITEMS_PER_BLOCK - 1) / CROSSOVER_ITEMS_PER_BLOCK, CROSSOVER_MUTATE_BLOCKS_MAX);
    u32 item_blocks_crossover = (ruleset_size + blocks_crossover - 1) / blocks_crossover;
    u32 blocks_mutate = min<u32>((ruleset_size + MUTATE_ITEMS_PER_BLOCK - 1) / MUTATE_ITEMS_PER_BLOCK, CROSSOVER_MUTATE_BLOCKS_MAX);
    u32 item_blocks_mutate = (ruleset_size + blocks_mutate - 1) / blocks_mutate;

    cudaDeviceSynchronize();

    // Run CROSSOVER_MUTATE_KERNELS_MAX in parallel
    for (u32 simulation_index = 0; simulation_index < POPULATION_SIZE; simulation_index++) {
        cudaStream_t stream = seeker->streams[simulation_index % CROSSOVER_MUTATE_KERNELS_MAX];
        simulation_t* simulation = &seeker->simulations[simulation_index];
        u32 rng_set_index = simulation_index % CROSSOVER_MUTATE_KERNELS_MAX;
        u64 seed = random_sample_u64();
        u32 source_ruleset_index_a = random_sample_u32(POPULATION_SELECTION);
        u32 source_ruleset_index_b = random_sample_u32(POPULATION_SELECTION);
        u8* source_ruleset_a = seeker->cpu_gpu_mutation_rulesets[source_ruleset_index_a];
        u8* source_ruleset_b = seeker->cpu_gpu_mutation_rulesets[source_ruleset_index_b];
        curandStatePhilox4_32_10_t* rng_set = seeker->cpu_gpu_rngs[rng_set_index];

        kernel_init_rngs<<<CROSSOVER_MUTATE_BLOCKS_MAX, THREADS_PER_BLOCK, 0, stream>>>(rng_set, seed);
        kernel_crossover<<<blocks_crossover, THREADS_PER_BLOCK, 0, stream>>>(rng_set, source_ruleset_a, source_ruleset_b, simulation->gpu_ruleset, item_blocks_crossover, ruleset_size);
        kernel_mutate<<<blocks_mutate, THREADS_PER_BLOCK, 0, stream>>>(rng_set, simulation->gpu_ruleset, item_blocks_mutate, ruleset_size);
    }

    if (CPU_VERIFY) {
        cudaDeviceSynchronize();

        for (simulation_t& simulation : seeker->simulations) {
            simulation_copy_ruleset_gpu_cpu(&simulation);
        }
    }

    cudaDeviceSynchronize();
}

void population_selection_mutation(array<simulation_t, POPULATION_SIZE>& simulations) {
    u32 ruleset_size = get_ruleset_size();

    std::vector<std::vector<u8>> rulesets;

    // Initialize rulesets
    while (rulesets.size() < POPULATION_SIZE) {
        std::vector<u8> ruleset = std::vector<u8>(ruleset_size, 0);
        rulesets.push_back(ruleset);
    }

    // Insert elites
    for (u32 i = 0; i < POPULATION_ELITES; i++) {
        memcpy(&rulesets[i][0], simulations[i].cpu_ruleset, ruleset_size * sizeof(u8));
    }

    // Crossover & mutate to fill up the rest of the population
    for (u32 i = POPULATION_ELITES; i < POPULATION_SIZE; i++) {
        u32 a_index = random_sample_u32(POPULATION_SELECTION);
        u32 b_index = random_sample_u32(POPULATION_SELECTION);
        u8* a = simulations[a_index].cpu_ruleset;
        u8* b = simulations[b_index].cpu_ruleset;
        u8* target = &rulesets[i][0];

        ruleset_crossover(a, b, target);
        ruleset_mutate(target);
    }

    // Store resulting rulesets to simulations and upload to the GPU
    for (u32 i = 0; i < POPULATION_SIZE; i++) {
        simulation_t* simulation = &simulations[i];

        memcpy(simulation->cpu_ruleset, &rulesets[i][0], ruleset_size * sizeof(u8));
        simulation_copy_ruleset_cpu_gpu(simulation);
    }

    // Wait for all rulesets to be copied
    for (simulation_t& simulation : simulations) {
        cudaStreamSynchronize(simulation.stream);
    }
}

void seeker_init(int argc, char **argv, seeker_t* seeker) {
    temp_storage_init(&seeker->temp_storage);

    for (cudaStream_t& stream : seeker->streams) {
        CHECK_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    for (u32 i = 0; i < CROSSOVER_MUTATE_KERNELS_MAX; i++) {
        CHECK_ERROR(cudaMalloc(&seeker->cpu_gpu_rngs[i], sizeof(curandStatePhilox4_32_10_t) * CROSSOVER_MUTATE_BLOCKS_MAX));
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

    if (argc >= 3) {
        printf("Loading initial grid: %s\n", argv[2]);
        grid_load(argv[2], &seeker->cpu_initial_grid[0]);
    } else {
        printf("No initial grid provided, using a random initial grid.\n");
        grid_init_random(&seeker->cpu_initial_grid[0]);
    }

    CHECK_ERROR(cudaMemcpy(seeker->gpu_initial_grid, &seeker->cpu_initial_grid[0], GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyHostToDevice));

    for (simulation_t& simulation : seeker->simulations) {
        simulation_init(&simulation, false, 0);
    }

    seeker->population_index = 0;

    printf("Initialized.\n");
}

void seeker_loop(seeker_t* seeker) {
    signal(SIGINT, sigint_handler_soft);

    while (!sigint_received) {
        // Load initial grid
        printf("Resetting grids...\n");
        for (simulation_t& simulation : seeker->simulations) {
            simulation_set_grid(&simulation, &seeker->cpu_initial_grid[0], seeker->gpu_initial_grid);
            cudaStreamSynchronize(simulation.stream);
        }
        printf("Grids reset finished.\n");

        if (seeker->population_index > 0) {
            printf("Performing selection & mutation...\n");
            /* population_selection_mutation(seeker->simulations); */
            seeker_crossover_mutation(seeker);
            printf("Selection & mutation finished.\n");

            cudaDeviceSynchronize();
        }

        for (simulation_t& simulation : seeker->simulations) {
            CHECK_ERROR(cudaStreamSynchronize(simulation.stream));
        }

        printf("Simulating population #%u...\n", seeker->population_index);
        population_simulate(seeker->simulations);
        printf("Finished simulating population #%u.\n", seeker->population_index);

        for (simulation_t& simulation : seeker->simulations) {
            CHECK_ERROR(cudaStreamSynchronize(simulation.stream));
        }

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

void seeker_output(seeker_t* seeker) {
    u32 rulesets_to_save;

    printf("How many rulesets would you like to save? [1]: ");

    if (scanf("%u", &rulesets_to_save) != 1) {
        rulesets_to_save = 1;
    }

    if (rulesets_to_save > POPULATION_SIZE) {
        rulesets_to_save = POPULATION_SIZE;
    }

    for (u32 i = 0; i < rulesets_to_save; i++) {
        simulation_t* simulation = &seeker->simulations[i];
        char filename[1024];

        snprintf(filename, 1024, "ruleset_%04u.rsr", i);
        simulation_ruleset_save(simulation, filename);
    }

    printf("%u rulesets saved.\n", rulesets_to_save);
}

void seeker_finalize(seeker_t* seeker) {
    for (simulation_t& simulation : seeker->simulations) {
        simulation_free(&simulation);
    }
}

int main_seek(int argc, char **argv) {
    initialize(argc, argv);

    seeker_t seeker;

    seeker_init(argc, argv, &seeker);
    seeker_loop(&seeker);
    seeker_output(&seeker);
    seeker_finalize(&seeker);

    return 0;
}

int main_simulate(int argc, char **argv) {
    init_draw(argc, argv, handle_keys, idle_func);
    initialize(argc, argv);

    return ui_loop();
}

int main(int argc, char **argv) {
    bool seek = argc >= 2 && strcmp(argv[1], "seek") == 0;
    bool simulate = argc >= 2 && strcmp(argv[1], "simulate") == 0;

    if (!seek && !simulate) {
        printf("Usage:\n");
        printf("%s seek GRID.rsg -- performs search for interesting rulesets\n", argv[0]);
        printf("%s simulate GRID.rsg RULESET.rsr -- performs visual simulation of an existing ruleset\n", argv[0]);
        exit(0);
    }

    print_configuration();
    random_init();

    if (PROMPT_TO_START) {
        printf("Press Enter to begin.");
        getchar();
    }

    if (seek) {
        return main_seek(argc, argv);
    }

    if (simulate) {
        return main_simulate(argc, argv);
    }

    return 0;
}
