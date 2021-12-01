#include <algorithm>
#include <signal.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <array>
#include <vector>
#include "common.h"
#include "util.cuh"
#include "draw.cuh"
#include "math.cuh"
#include "simulation.cuh"

using namespace std;

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
        printf("`POPULATION_SELECTION` may not exceed `POPULATION_SIZE`.");
    }

    if (POPULATION_ELITES > POPULATION_SIZE) {
        printf("`POPULATION_ELITES` may not exceed `POPULATION_SIZE`.");
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

/* funkce zajistujici aktualizaci simulace - verze pro CPU
 *   in_grid - vstupni simulacni mrizka
 *   out_grid - vystupni simulacni mrizka
 *   width - sirka simulacni mrizky
 *   height - vyska simulacni mrizky
 *   fit_cells_block_sum - pointer to a single counter of fit cells
 */
__host__ void cpu_simulate_step(u8* in_grid, u8* out_grid, u8* ruleset, u32* fit_cells_block_sum) {
    *fit_cells_block_sum = 0;

    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            bool fit = update_cell(in_grid, out_grid, ruleset, x, y);
            *fit_cells_block_sum += (u32) fit;
        }
    }
}

/* funkce zajistujici aktualizaci simulace - verze pro CPU
 *   in_grid - vstupni simulacni mrizka
 *   out_grid - vystupni simulacni mrizka
 *   width - sirka simulacni mrizky
 *   height - vyska simulacni mrizky
 *   fit_cells_block_sum - pointer to counters of fit cells, one per block
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

    // initialize fit_cells_block_sum to 0
    fit_cells_block_sums[block_index_1d] = 0;

    __syncthreads();

    atomicAdd(&fit_cells_block_sums[block_index_1d], (u32) fit);
}

// Simulates a single iteration
void simulate_step(simulation_t* simulation, bool async) {
    const i32 STEPS = 1;
    // might as well measure the time since we have to wait for the result anyway
    async &= !CPU_VERIFY;
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
            gpu_simulate_step_kernel_shared<<<GRID_DIM_BLOCKS, GRID_DIM_THREADS, 0, simulation->stream>>>(gpu_grid_states_1, gpu_grid_states_2, simulation->gpu_ruleset, simulation->gpu_fit_cells_block_sums);
        } else {
            gpu_simulate_step_kernel_noshared<<<GRID_DIM_BLOCKS, GRID_DIM_THREADS, 0, simulation->stream>>>(gpu_grid_states_1, gpu_grid_states_2, simulation->gpu_ruleset, simulation->gpu_fit_cells_block_sums);
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

    cudaMemcpyAsync(simulation->cpu_grid_states_tmp, gpu_grid_states_1, GRID_AREA_WITH_PITCH * sizeof(u8), cudaMemcpyDeviceToHost, simulation->stream);

    simulation_collect_fit_cells_block_sums_async(simulation);
    cudaStreamSynchronize(simulation->stream);
    simulation_reduce_fit_cells(simulation);

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
    simulate_step(&preview_simulation, false);
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
        for (simulation_t& simulation : simulations) {
            simulate_step(&simulation, true);

            if (iteration >= FITNESS_EVAL_FROM) {
                simulation_collect_fit_cells_block_sums_async(&simulation);
            }

            check_sigint(&sigint_acknowledged);
        }

        if (iteration >= FITNESS_EVAL_FROM) {
            // Wait for the iteration to finish
            for (simulation_t& simulation : simulations) {
                check_sigint(&sigint_acknowledged);

                CHECK_ERROR(cudaStreamSynchronize(simulation.stream));
            }

            // Cumulate error
            for (simulation_t& simulation : simulations) {
                simulation_reduce_fit_cells(&simulation);
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

bool simulation_cumulative_error_comparator(simulation_t& a, simulation_t& b) {
    return a.cumulative_error < b.cumulative_error;
}

void population_order(array<simulation_t, POPULATION_SIZE>& simulations) {
    for (simulation_t& simulation : simulations) {
        simulation_copy_ruleset_gpu_cpu(&simulation);
    }

    // Wait for all rulesets to be copied
    for (simulation_t& simulation : simulations) {
        cudaStreamSynchronize(simulation.stream);
    }

    // Order simulations by cumulative_error
    sort(simulations.begin(), simulations.end(), simulation_cumulative_error_comparator);
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

int main_seek(int argc, char **argv) {
    initialize(argc, argv);
    signal(SIGINT, sigint_handler_soft);

    vector<u8> initial_grid = vector<u8>(GRID_AREA_WITH_PITCH, 0);

    if (argc >= 3) {
        printf("Loading initial grid: %s\n", argv[2]);
        grid_load(argv[2], &initial_grid[0]);
    } else {
        printf("No initial grid provided, using a random initial grid.\n");
        grid_init_random(&initial_grid[0]);
    }

    array<simulation_t, POPULATION_SIZE> simulations;

    for (simulation_t& simulation : simulations) {
        simulation_init(&simulation, false, 0);
    }

    printf("Initialized.\n");

    i32 population_index = 0;

    while (!sigint_received) {
        printf("Resetting grids...\n");
        // Load initial grid
        for (simulation_t& simulation : simulations) {
            memcpy(simulation.cpu_grid_states_1, &initial_grid[0], GRID_AREA_WITH_PITCH * sizeof(u8));
            simulation_copy_grid_cpu_gpu(&simulation);
            cudaStreamSynchronize(simulation.stream);
        }
        printf("Grids reset finished.\n");

        if (population_index > 0) {
            printf("Performing selection & mutation...\n");
            population_selection_mutation(simulations);
            printf("Selection & mutation finished.\n");
        }

        printf("Simulating population #%u...", population_index);
        population_simulate(simulations);
        printf("Finished simulating population #%u.\n", population_index);
        printf("Ordering population...\n");
        population_order(simulations);
        printf("Population ordered.\n");

        {
            printf("cumulative_error:\n");

            for (simulation_t& simulation : simulations) {
                CHECK_ERROR(cudaStreamSynchronize(simulation.stream));
            }

            for (simulation_t& simulation : simulations) {
                simulation_cumulative_error_normalize(&simulation);
            }

            for (simulation_t& simulation : simulations) {
                printf("%f, ", simulation.cumulative_error);
            }

            printf("\n");
        }

        population_index++;

        #ifdef EXIT_AFTER_POPULATIONS
        if (population_index >= EXIT_AFTER_POPULATIONS) {
            break;
        }
        #endif
    }

    printf("Simulations finished.\n");
    signal(SIGINT, sigint_handler_abort);

    u32 rulesets_to_save;

    printf("How many rulesets would you like to save? [1]: ");

    if (scanf("%u", &rulesets_to_save) != 1) {
        rulesets_to_save = 1;
    }

    if (rulesets_to_save > POPULATION_SIZE) {
        rulesets_to_save = POPULATION_SIZE;
    }

    for (u32 i = 0; i < rulesets_to_save; i++) {
        simulation_t* simulation = &simulations[i];
        char filename[1024];

        snprintf(filename, 1024, "ruleset_%04u.rsr", i);
        simulation_ruleset_save(simulation, filename);
    }

    printf("%u rulesets saved.\n", rulesets_to_save);

    for (simulation_t& simulation : simulations) {
        simulation_free(&simulation);
    }

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