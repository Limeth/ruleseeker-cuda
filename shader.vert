// (1) Prepended version directive
// (2) Prepended config.h header
layout (location = 0) in uint in_cell_state_current;
layout (location = 1) in uint in_cell_state_previous;

layout (location = 0) out flat uint out_cell_state_current;
layout (location = 1) out flat uint out_cell_state_previous;

void main() {
#if GRID_GEOMETRY == GRID_GEOMETRY_SQUARE
    vec2[4] positions = {
        vec2(0, 0),
        vec2(1, 0),
        vec2(1, 1),
        vec2(0, 1),
    };
#elif GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE
    // TODO
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    // TODO
#endif

    ivec2 cell = ivec2(gl_InstanceID % GRID_WIDTH, gl_InstanceID / GRID_WIDTH);
    vec2 position = (positions[gl_VertexID] + cell) / vec2(GRID_WIDTH, GRID_HEIGHT);
    position = position * 2.0 - 1.0;
    gl_Position = vec4(position, 1.0, 1.0);

    out_cell_state_current = in_cell_state_current;
    out_cell_state_previous = in_cell_state_previous;
}
