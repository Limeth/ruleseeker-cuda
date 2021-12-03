.PHONY = release debug all clean

CC = nvcc

LINKER_FLAGS := -lGLEW -lglut -lGL -lpng -lbsd -lboost_system
COMPILER_FLAGS := -I/usr/local/cuda/include -O3 -DPNG_NO_SETJMP

SRC_DIR = src
SRC := $(wildcard ${SRC_DIR}/*.cu)
HEADERS := $(wildcard ${SRC_DIR}/*.cuh) $(wildcard ${SRC_DIR}/*.h)
OBJ := $(SRC:.cu=.o)

release: COMPILER_FLAGS += -DNDEBUG
release: BIN := build-release
release: all

debug: COMPILER_FLAGS += -g -G
debug: BIN := build-debug
debug: all

all: ${SRC_DIR}/main.o
	@echo "Linking ${BIN}"
	./include-shaders.sh
	${CC} ${LINKER_FLAGS} $< -o ${BIN}

# %.o: %.cu
${SRC_DIR}/main.o: ${SRC_DIR}/main.cu ${HEADERS}
	@echo "Compiling $@"
	${CC} -c $< -o $@ ${COMPILER_FLAGS}

clean:
	@echo "Cleaning up"
	rm -rvf $(OBJ) build-release build-debug
