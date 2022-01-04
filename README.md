# Compiling
Requirements:
* libglew-dev: For initializing OpenGL
* libglfw3-dev: For creating and drawing to a window
* libpng-dev: For exporting frames
* libbsd-dev: For random number generation
* libboost1.71-dev, libboost-system1.71-dev: For sampling of variables with the binomial distribution
* Cuda Toolkit 11.5 (not tested with other versions): For compute tasks!

To compile, ensure dependencies are installed on the system and simply run:
```sh
make
```

This will have generated a `build-release` executable binary.

# Usage

1. First, the application must be configured in `src/config.h`.
2. Then, make sure to compile the application, so that the configuration is applied.
3. Finally, run the compiled application in one of the following ways:
    1. `./build-release edit GRID.rsg [-r]`
    -- Opens up an interactive editor of the grid, which allows the user to create an initial grid state
       to be used by other commands.
    2. `./build-release seek GRID.rsg [RULESET.rsr]`
    -- Performs a search for the best rulesets using a genetic algorithm, as defined in the configuration.
       If a ruleset is specified, it is used, otherwise, begins with random rulesets.
       Press Ctrl-C once to end the search and save the rulesets to `ruleset_i.rsr`.
    3. `./build-release show GRID.rsg [RULESET.rsr]`
    -- Displays a simulation with the given initial grid and provided ruleset, if specified, or a random ruleset, if not.
