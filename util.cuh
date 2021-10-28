#pragma once
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

// funkce pro osetreni chyb
static inline void check_error(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(error), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CHECK_ERROR(error) (check_error(error, __FILE__, __LINE__))

#ifdef NDEBUG
    #define IS_DEBUG false
#else
    #define IS_DEBUG true
#endif
