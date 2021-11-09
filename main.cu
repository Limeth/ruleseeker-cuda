#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "config.h"
#include "util.cuh"
#include "draw.cuh"
#include "math.cuh"

using namespace std;

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
    printf("\n");
}

__device__ u8* device_gpu_ruleset;
u8* gpu_ruleset = NULL;
u8* cpu_ruleset = NULL;

__inline__ __host__ __device__ u8* get_ruleset() {
#ifdef __CUDA_ARCH__
    return device_gpu_ruleset;
#else
    return cpu_ruleset;
#endif
}

// nahrazeno CUDA zdroji
/* __device__ u8* gpu_grid_states_1 = NULL; */
/* __device__ u8* gpu_grid_states_2 = NULL; */

u8 *cpu_grid_states_1 = NULL;
u8 *cpu_grid_states_2 = NULL;
u8 *cpu_grid_states_tmp = NULL;

// udalosti pro mereni casu v CUDA
cudaEvent_t start, stop;

__inline__ __host__ __device__ int get_cell_index(int width, int height, int x, int y) {
    x = mod(x, width);
    y = mod(y, height);

    return x + y * width;
}

__host__ __device__ u8 get_next_state(u8 current_state, u8* neighbours) {
    u8* ruleset = get_ruleset();

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

__host__ __device__ u8 update_cell(u8* in, u8* out, int width, int height, int x, int y) {
    int cellID = get_cell_index(width, height, x, y);
    u8 currentState = in[cellID];
    i32vec2 neighbours[CELL_NEIGHBOURHOOD_SIZE];
    u8 state_count[CELL_STATES] = { 0 };

    get_neighbours(x, y, neighbours);

    for (u32 neighbour_index = 0; neighbour_index < CELL_NEIGHBOURHOOD_SIZE; neighbour_index++) {
        i32vec2 neighbour = neighbours[neighbour_index];
        i32 abs_x = x + neighbour.x;
        i32 abs_y = y + neighbour.y;
        i32 neighbour_cell_index = get_cell_index(width, height, abs_x, abs_y);

        state_count[in[neighbour_cell_index]] += 1;
    }

    int nextState = get_next_state(currentState, state_count);
    out[cellID] = nextState;

    return nextState;
}

/* funkce zajistujici aktualizaci simulace - verze pro CPU
 *  in_grid - vstupni simulacni mrizka
 *  out_grid - vystupni simulacni mrizka
 *  width - sirka simulacni mrizky
 *  height - vyska simulacni mrizky
 */
__host__ void cpu_simulate_step(u8* in_grid, u8* out_grid) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            update_cell(in_grid, out_grid, GRID_WIDTH, GRID_HEIGHT, x, y);
        }
    }
}

__global__ void gpu_simulate_step_kernel(u8* in_grid, u8* out_grid) {
    int x = threadIdx.x + blockIdx.x * BLOCK_LENGTH;
    int y = threadIdx.y + blockIdx.y * BLOCK_LENGTH;

    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        u8 newValue = update_cell(in_grid, out_grid, GRID_WIDTH, GRID_HEIGHT, x, y);
    }
}

__device__ void gpu_simulate_step(u8* in_grid, u8* out_grid) {
    __syncthreads();

    // grid and block dimensions
    dim3 blocks(GRID_WIDTH_IN_BLOCKS, GRID_HEIGHT_IN_BLOCKS);
    dim3 threads(BLOCK_LENGTH, BLOCK_LENGTH);

    gpu_simulate_step_kernel<<<blocks, threads>>>(in_grid, out_grid);

    __syncthreads();
    cudaDeviceSynchronize();
    __syncthreads();
}

__inline__ __host__ __device__ void simulate_step(u8* in_grid, u8* out_grid) {
#ifdef __CUDA_ARCH__
    gpu_simulate_step(in_grid, out_grid);
#else
    cpu_simulate_step(in_grid, out_grid);
#endif
}

__device__ __host__ void simulate_steps(u8* in, u8* out, u32 n) {
    for (u32 i = 0; i < n; i++) {
        simulate_step(in, out);

        // swap pointers
        u8* tmp = in;
        in = out;
        out = tmp;
    }
}

__global__ void gpu_kernel_simulate(u8* in_grid, u8* out_grid) {
    simulate_step(in_grid, out_grid);
}

__host__ void cpu_kernel_simulate(u8* in_grid, u8* out_grid) {
    simulate_step(in_grid, out_grid);
}


void simulate_multiple_steps() {
    // ulozeni pocatecniho casu
    CHECK_ERROR(cudaEventRecord(start, 0));

    // grid and block dimensions
    dim3 blocks(GRID_WIDTH_IN_BLOCKS, GRID_HEIGHT_IN_BLOCKS);
    dim3 threads(BLOCK_LENGTH, BLOCK_LENGTH);

    CHECK_ERROR(cudaGraphicsMapResources(1, &gpu_cuda_grid_states_1, 0));
    CHECK_ERROR(cudaGraphicsMapResources(1, &gpu_cuda_grid_states_2, 0));

    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_1, NULL, gpu_cuda_grid_states_1));
    CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_2, NULL, gpu_cuda_grid_states_2));

    // aktualizace simulace + vygenerovani bitmapy pro zobrazeni stavu simulace
    gpu_kernel_simulate<<<blocks,threads>>>(gpu_grid_states_1, gpu_grid_states_2);

    swap(gpu_vbo_grid_states_1, gpu_vbo_grid_states_2);
    swap(gpu_cuda_grid_states_1, gpu_cuda_grid_states_2);
    swap(gpu_grid_states_1, gpu_grid_states_2);

    // ulozeni casu ukonceni simulace
    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;

    // vypis casu simulace
    CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Update: %f ms\n", elapsedTime);

#if CPU_VERIFY
    printf("Verifying on the CPU...\n");
    // krok simulace life game na CPU
    cpu_kernel_simulate(cpu_grid_states_1, cpu_grid_states_2);
    swap(cpu_grid_states_1, cpu_grid_states_2);

    cudaMemcpy(cpu_grid_states_tmp, gpu_grid_states_1, GRID_AREA * sizeof(u8), cudaMemcpyDeviceToHost);

    int diffs = 0;

    // porovnani vysledku CPU simulace a GPU simulace
    for (i32 y = 0; y < GRID_HEIGHT; y++) {
        for (i32 x = 0; x < GRID_WIDTH; x++) {
            i32 threadID = x + y * GRID_WIDTH;

            if (cpu_grid_states_1[threadID] != cpu_grid_states_tmp[threadID]) {
                diffs++;
            }
        }
    }

    if(diffs != 0)
        std::cout << "CHYBA: " << diffs << " rozdily mezi CPU & GPU simulacni mrizkou" << std::endl;
#endif

    CHECK_ERROR(cudaGraphicsUnmapResources(1, &gpu_cuda_grid_states_1, 0));
    CHECK_ERROR(cudaGraphicsUnmapResources(1, &gpu_cuda_grid_states_2, 0));

}


// funkce pro spusteni kernelu + priprava potrebnych dat a struktur
void callKernelCUDA(void) {
    // ulozeni pocatecniho casu
    CHECK_ERROR(cudaEventRecord(start, 0));

    // grid and block dimensions
    dim3 blocks(GRID_WIDTH_IN_BLOCKS, GRID_HEIGHT_IN_BLOCKS);
    dim3 threads(BLOCK_LENGTH, BLOCK_LENGTH);

    CHECK_ERROR(cudaGraphicsMapResources(1, &gpu_cuda_grid_states_1, 0));
    CHECK_ERROR(cudaGraphicsMapResources(1, &gpu_cuda_grid_states_2, 0));

    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_1, NULL, gpu_cuda_grid_states_1));
    CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_2, NULL, gpu_cuda_grid_states_2));

    // aktualizace simulace + vygenerovani bitmapy pro zobrazeni stavu simulace
    gpu_kernel_simulate<<<blocks,threads>>>(gpu_grid_states_1, gpu_grid_states_2);

    swap(gpu_vbo_grid_states_1, gpu_vbo_grid_states_2);
    swap(gpu_cuda_grid_states_1, gpu_cuda_grid_states_2);
    swap(gpu_grid_states_1, gpu_grid_states_2);

    // ulozeni casu ukonceni simulace
    CHECK_ERROR(cudaEventRecord(stop, 0));
    CHECK_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;

    // vypis casu simulace
    CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Update: %f ms\n", elapsedTime);

#if CPU_VERIFY
    printf("Verifying on the CPU...\n");
    // krok simulace life game na CPU
    cpu_kernel_simulate(cpu_grid_states_1, cpu_grid_states_2);
    swap(cpu_grid_states_1, cpu_grid_states_2);

    cudaMemcpy(cpu_grid_states_tmp, gpu_grid_states_1, GRID_AREA * sizeof(u8), cudaMemcpyDeviceToHost);

    int diffs = 0;

    // porovnani vysledku CPU simulace a GPU simulace
    for (i32 y = 0; y < GRID_HEIGHT; y++) {
        for (i32 x = 0; x < GRID_WIDTH; x++) {
            i32 threadID = x + y * GRID_WIDTH;

            if (cpu_grid_states_1[threadID] != cpu_grid_states_tmp[threadID]) {
                diffs++;
            }
        }
    }

    if(diffs != 0)
        std::cout << "CHYBA: " << diffs << " rozdily mezi CPU & GPU simulacni mrizkou" << std::endl;
#endif

    CHECK_ERROR(cudaGraphicsUnmapResources(1, &gpu_cuda_grid_states_1, 0));
    CHECK_ERROR(cudaGraphicsUnmapResources(1, &gpu_cuda_grid_states_2, 0));
}

// called every frame
void idle_func() {
    callKernelCUDA();
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

// Writes a ruleset to a file
void ruleset_save(u8* ruleset, char* filename) {
    FILE* file = fopen(filename, "wb");

    if (get_ruleset_size() != fwrite(ruleset, sizeof(u8), get_ruleset_size(), file)) {
        fprintf(stderr, "Failed to write a ruleset.");
        exit(1);
    }

    fclose(file);
}

// Loads a ruleset from a file to a pre-allocated buffer
void ruleset_load(u8* ruleset, char* filename) {
    FILE* file = fopen(filename, "rb");

    if (get_ruleset_size() != fread(ruleset, sizeof(u8), get_ruleset_size(), file)) {
        fprintf(stderr, "Failed to load a ruleset.");
        exit(1);
    }

    fclose(file);
}

void ruleset_load_alloc(u8** ruleset, char* filename) {
    *ruleset = (u8*) calloc(get_ruleset_size(), sizeof(u8));
    ruleset_load(*ruleset, filename);
}

// inicializace CUDA - alokace potrebnych dat a vygenerovani pocatecniho stavu lifu
void initialize(int argc, char **argv) {
    init_draw(argc, argv, handle_keys, idle_func);

    // alokovani mista pro bitmapu na GPU
    CHECK_ERROR(cudaMalloc((void**) &(gpu_ruleset), get_ruleset_size() * sizeof(u8)));

    cpu_ruleset = (u8*) calloc(get_ruleset_size(), sizeof(u8));
    cpu_grid_states_1 = (u8*) calloc(GRID_AREA, sizeof(u8));
    cpu_grid_states_2 = (u8*) calloc(GRID_AREA, sizeof(u8));
    cpu_grid_states_tmp = (u8*) calloc(GRID_AREA, sizeof(u8));

    srand(0);

    /* for (int i = 0; i < GRID_AREA; i++) { */
    /*     cpu_grid_states_1[i] = (u8) (rand() % CELL_STATES); */
    /* } */
    cpu_grid_states_1[(GRID_HEIGHT / 2) * GRID_WIDTH + GRID_WIDTH / 2] = 1;

    CHECK_ERROR(cudaGraphicsMapResources(1, &gpu_cuda_grid_states_1, 0));
    CHECK_ERROR(cudaGraphicsMapResources(1, &gpu_cuda_grid_states_2, 0));

    u8* gpu_grid_states_1 = NULL;
    u8* gpu_grid_states_2 = NULL;

    CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_1, NULL, gpu_cuda_grid_states_1));
    CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &gpu_grid_states_2, NULL, gpu_cuda_grid_states_2));

    // prekopirovani pocatecniho stavu do GPU
    cudaMemcpy(gpu_grid_states_1, cpu_grid_states_1, GRID_AREA * sizeof(u8), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_grid_states_2, cpu_grid_states_1, GRID_AREA * sizeof(u8), cudaMemcpyHostToDevice);

    CHECK_ERROR(cudaGraphicsUnmapResources(1, &gpu_cuda_grid_states_1, 0));
    CHECK_ERROR(cudaGraphicsUnmapResources(1, &gpu_cuda_grid_states_2, 0));

    // Initialize ruleset. Keep first rule as 0.
    for (i32 i = 1; i < get_ruleset_size(); i++) {
        cpu_ruleset[i] = (u8) (rand() % CELL_STATES);
    }

    CHECK_ERROR(cudaMemcpy(gpu_ruleset, cpu_ruleset, get_ruleset_size() * sizeof(u8), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpyToSymbol(device_gpu_ruleset, &gpu_ruleset, sizeof(u8*)));

    // vytvoreni struktur udalosti pro mereni casu
    CHECK_ERROR(cudaEventCreate( &start ));
    CHECK_ERROR(cudaEventCreate( &stop ));
}

// funkce volana pri ukonceni aplikace, uvolni vsechy prostredky alokovane v CUDA 
void finalize(void) {
    // uvolneni bitmapy - na CPU i GPU
    cudaFree(gpu_ruleset);

    // uvolneni simulacnich mrizek pro CPU variantu lifu
    free(cpu_ruleset);
    free(cpu_grid_states_1);
    free(cpu_grid_states_2);
    free(cpu_grid_states_tmp);

    // zruseni struktur udalosti
    CHECK_ERROR(cudaEventDestroy( start ));
    CHECK_ERROR(cudaEventDestroy( stop ));

    finalize_draw();
}

int main_simulate(int argc, char **argv) {
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

    if (PROMPT_TO_START) {
        printf("Press Enter to begin.");
        getchar();
    }

    if (simulate) {
        return main_simulate(argc, argv);
    }

    return 0;
}
