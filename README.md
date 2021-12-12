# Compiling
Requirements:
* libglew-dev: For initializing OpenGL
* freeglfw3-dev: For creating and drawing to a window
* libpng-dev: For exporting frames
* libbsd-dev: For random number generation
* libboost1.71-dev: For sampling of variables with the binomial distribution
* Cuda Toolkit 11.5 (not tested with other versions): For compute tasks!

To compile, ensure dependencies are installed on the system and simply run:
```sh
make
```

This will have generated a `build-release` executable binary.
