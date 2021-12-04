#pragma once
#include <bsd/stdlib.h>
#include <stdio.h>
#include <iostream>

#define i8 char
#define i16 short
#define i32 int
#define i64 long
#define u8 unsigned char
#define u16 unsigned short
#define u32 unsigned int
#define u64 unsigned long
#define f32 float
#define f64 double

#define i8vec1  char1
#define i16vec1 short1
#define i32vec1 int1
#define i64vec1 long1
#define u8vec1  uchar1
#define u16vec1 ushort1
#define u32vec1 uint1
#define u64vec1 ulong1
#define f32vec1 float1
#define f64vec1 double1

#define i8vec2  char2
#define i16vec2 short2
#define i32vec2 int2
#define i64vec2 long2
#define u8vec2  uchar2
#define u16vec2 ushort2
#define u32vec2 uint2
#define u64vec2 ulong2
#define f32vec2 float2
#define f64vec2 double2

#define i8vec3  char3
#define i16vec3 short3
#define i32vec3 int3
#define i64vec3 long3
#define u8vec3  uchar3
#define u16vec3 ushort3
#define u32vec3 uint3
#define u64vec3 ulong3
#define f32vec3 float3
#define f64vec3 double3

#define i8vec4  char4
#define i16vec4 short4
#define i32vec4 int4
#define i64vec4 long4
#define u8vec4  uchar4
#define u16vec4 ushort4
#define u32vec4 uint4
#define u64vec4 ulong4
#define f32vec4 float4
#define f64vec4 double4

#define make_i8vec1(...)  make_char1(__VA_ARGS__)
#define make_i16vec1(...) make_short1(__VA_ARGS__)
#define make_i32vec1(...) make_int1(__VA_ARGS__)
#define make_i64vec1(...) make_long1(__VA_ARGS__)
#define make_u8vec1(...)  make_uchar1(__VA_ARGS__)
#define make_u16vec1(...) make_ushort1(__VA_ARGS__)
#define make_u32vec1(...) make_uint1(__VA_ARGS__)
#define make_u64vec1(...) make_ulong1(__VA_ARGS__)
#define make_f32vec1(...) make_float1(__VA_ARGS__)
#define make_f64vec1(...) make_double1(__VA_ARGS__)

#define make_i8vec2(...)  make_char2(__VA_ARGS__)
#define make_i16vec2(...) make_short2(__VA_ARGS__)
#define make_i32vec2(...) make_int2(__VA_ARGS__)
#define make_i64vec2(...) make_long2(__VA_ARGS__)
#define make_u8vec2(...)  make_uchar2(__VA_ARGS__)
#define make_u16vec2(...) make_ushort2(__VA_ARGS__)
#define make_u32vec2(...) make_uint2(__VA_ARGS__)
#define make_u64vec2(...) make_ulong2(__VA_ARGS__)
#define make_f32vec2(...) make_float2(__VA_ARGS__)
#define make_f64vec2(...) make_double2(__VA_ARGS__)

#define make_i8vec3(...)  make_char3(__VA_ARGS__)
#define make_i16vec3(...) make_short3(__VA_ARGS__)
#define make_i32vec3(...) make_int3(__VA_ARGS__)
#define make_i64vec3(...) make_long3(__VA_ARGS__)
#define make_u8vec3(...)  make_uchar3(__VA_ARGS__)
#define make_u16vec3(...) make_ushort3(__VA_ARGS__)
#define make_u32vec3(...) make_uint3(__VA_ARGS__)
#define make_u64vec3(...) make_ulong3(__VA_ARGS__)
#define make_f32vec3(...) make_float3(__VA_ARGS__)
#define make_f64vec3(...) make_double3(__VA_ARGS__)

#define make_i8vec4(...)  make_char4(__VA_ARGS__)
#define make_i16vec4(...) make_short4(__VA_ARGS__)
#define make_i32vec4(...) make_int4(__VA_ARGS__)
#define make_i64vec4(...) make_long4(__VA_ARGS__)
#define make_u8vec4(...)  make_uchar4(__VA_ARGS__)
#define make_u16vec4(...) make_ushort4(__VA_ARGS__)
#define make_u32vec4(...) make_uint4(__VA_ARGS__)
#define make_u64vec4(...) make_ulong4(__VA_ARGS__)
#define make_f32vec4(...) make_float4(__VA_ARGS__)
#define make_f64vec4(...) make_double4(__VA_ARGS__)

// funkce pro osetreni chyb
static inline __host__ __device__ void check_error(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
#ifdef __CUDA_ARCH__
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        assert(0);
#else
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
#endif
    }
}

#define CHECK_ERROR(error) (check_error(error, __FILE__, __LINE__))

#ifdef NDEBUG
    #define IS_DEBUG false
#else
    #define IS_DEBUG true
#endif

#define IS_DEBUG_THREAD(thread_x, thread_y, block_x, block_y) (threadIdx.x == thread_x && threadIdx.y == thread_y && blockIdx.x == block_x && blockIdx.y == block_y)

#define DEBUG_THREAD(thread_x, thread_y, block_x, block_y, ...)            \
    do {                                                                   \
        if (IS_DEBUG_THREAD(thread_x, thread_y, block_x, block_y)) {       \
            printf("DEBUG_THREAD in %s on line %d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                                           \
        }                                                                  \
    } while(false);

// Callback function for printing debug statements
void APIENTRY gl_debug_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *msg, const void *data) {
    const char* _source;
    const char* _type;
    const char* _severity;

    switch (source) {
        case GL_DEBUG_SOURCE_API:
        _source = "API";
        break;

        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        _source = "WINDOW SYSTEM";
        break;

        case GL_DEBUG_SOURCE_SHADER_COMPILER:
        _source = "SHADER COMPILER";
        break;

        case GL_DEBUG_SOURCE_THIRD_PARTY:
        _source = "THIRD PARTY";
        break;

        case GL_DEBUG_SOURCE_APPLICATION:
        _source = "APPLICATION";
        break;

        case GL_DEBUG_SOURCE_OTHER:
        _source = "UNKNOWN";
        break;

        default:
        _source = "UNKNOWN";
        break;
    }

    switch (type) {
        case GL_DEBUG_TYPE_ERROR:
        _type = "ERROR";
        break;

        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        _type = "DEPRECATED BEHAVIOR";
        break;

        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        _type = "UDEFINED BEHAVIOR";
        break;

        case GL_DEBUG_TYPE_PORTABILITY:
        _type = "PORTABILITY";
        break;

        case GL_DEBUG_TYPE_PERFORMANCE:
        _type = "PERFORMANCE";
        break;

        case GL_DEBUG_TYPE_OTHER:
        _type = "OTHER";
        break;

        case GL_DEBUG_TYPE_MARKER:
        _type = "MARKER";
        break;

        default:
        _type = "UNKNOWN";
        break;
    }

    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:
        _severity = "HIGH";
        break;

        case GL_DEBUG_SEVERITY_MEDIUM:
        _severity = "MEDIUM";
        break;

        case GL_DEBUG_SEVERITY_LOW:
        _severity = "LOW";
        break;

        case GL_DEBUG_SEVERITY_NOTIFICATION:
        _severity = "NOTIFICATION";
        break;

        default:
        _severity = "UNKNOWN";
        break;
    }

    printf("\x1B[1;30m[OpenGL] %d: %s of %s severity, raised from %s: %s\x1B[0m\n",
            id, _type, _severity, _source, msg);
}

int msleep(long msec) {
    struct timespec ts;
    int res;

    if (msec < 0) {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

cudaError_t cudaCallocAsync(void** ptr, size_t size, cudaStream_t stream) {
    cudaError_t error;
    error = cudaMalloc(ptr, size);
    if (error) {
        return error;
    }
    return cudaMemsetAsync(*ptr, 0, size, stream);
}

cudaError_t cudaCallocHostAsync(void** ptr, size_t size, cudaStream_t stream) {
    cudaError_t error;
    error = cudaMallocHost(ptr, size);
    if (error) {
        return error;
    }
    return cudaMemsetAsync(*ptr, 0, size, stream);
}

// Write bytes to a file
bool file_write(char* filename, void* buffer, u32 buffer_len) {
    FILE* file = fopen(filename, "wb");

    if (buffer_len != fwrite(buffer, sizeof(u8), buffer_len, file)) {
        return false;
    }

    fclose(file);
    return true;
}

// Load bytes from a file
bool file_load(char* filename, u8* buffer, u32 buffer_len) {
    FILE* file = fopen(filename, "rb");

    if (!file) {
        return false;
    }

    if (buffer_len != fread(buffer, sizeof(u8), buffer_len, file)) {
        return false;
    }

    fclose(file);
    return true;
}

// Writes a 2d buffer with pitch to a file
bool file_write_2d_pitch(char* filename, u8* buffer, u32 width, u32 height, u32 pitch) {
    FILE* file = fopen(filename, "wb");

    for (i32 y = 0; y < height; y++) {
        if (width != fwrite(buffer, sizeof(u8), width, file)) {
            return false;
        }

        buffer += pitch;
    }

    fclose(file);
    return true;
}

// Writes a 2d buffer with pitch to a file
bool file_load_2d_pitch(char* filename, u8* buffer, u32 width, u32 height, u32 pitch) {
    FILE* file = fopen(filename, "rb");

    if (!file) {
        return false;
    }

    for (i32 y = 0; y < height; y++) {
        if (width != fread(buffer, sizeof(u8), width, file)) {
            return false;
        }

        buffer += pitch;
    }

    fclose(file);
    return true;
}

void random_init() {
    if (DETERMINISTIC_RANDOMNESS) {
        srand(0);
    }
}

u32 random_sample_u32(u32 upper_bound_exclusive) {
    if (DETERMINISTIC_RANDOMNESS) {
        // not uniform, but oh well
        return ((u32) rand()) % upper_bound_exclusive;
    } else {
        return arc4random_uniform(upper_bound_exclusive);
    }
}

u64 random_sample_u64() {
    if (DETERMINISTIC_RANDOMNESS) {
        // not uniform, but oh well
        return rand();
    } else {
        u64 result;
        arc4random_buf(&result, sizeof(u64));
        return result;
    }
}

f32 random_sample_f32_normalized() {
    return ((f32) random_sample_u32(RAND_MAX) / (f32) (RAND_MAX - 1));
}

f32 random_sample_f32(f32 upper_bound_inclusive) {
    return random_sample_f32_normalized() * upper_bound_inclusive;
}

f64 random_sample_f64_normalized() {
    return ((f64) random_sample_u32(RAND_MAX) / (f64) (RAND_MAX - 1));
}

f64 random_sample_f64(f64 upper_bound_inclusive) {
    return random_sample_f64_normalized() * upper_bound_inclusive;
}

// Temporary storage for CUB-related routines
typedef struct {
    u8* allocation;
    size_t size;
} temp_storage_t;

void temp_storage_init(temp_storage_t* temp_storage) {
    temp_storage->allocation = NULL;
    temp_storage->size = 0;
}

void temp_storage_ensure(temp_storage_t* temp_storage, size_t size, cudaStream_t stream) {
    if (size <= temp_storage->size) {
        return; // temp storage large enough, no need for realloc
    }

    if (temp_storage->allocation) {
        CHECK_ERROR(cudaFreeAsync(temp_storage->allocation, stream));
    }

    CHECK_ERROR(cudaMallocAsync(&temp_storage->allocation, size, stream));

    temp_storage->size = size;
}
