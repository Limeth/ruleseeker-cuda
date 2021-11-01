// (1) Prepended version directive
// (2) Prepended config.h header
#define TAU 6.28318530717958647692528676655900576839433879875021

layout (location = 0) in uint in_cell_state_current;
layout (location = 1) in uint in_cell_state_previous;

layout (location = 0) out flat uint out_cell_state_current;
layout (location = 1) out flat uint out_cell_state_previous;
layout (location = 2) out flat vec3 out_color;

uniform uvec2 window_resolution;

// All components are in the range [0…1], including hue.
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    ivec2 cell = ivec2(gl_InstanceID % GRID_WIDTH, gl_InstanceID / GRID_WIDTH);

#if GRID_GEOMETRY == GRID_GEOMETRY_SQUARE
    vec2[4] offsets_vertex = {
        vec2(0, 0),
        vec2(1, 0),
        vec2(1, 1),
        vec2(0, 1),
    };
    vec2 viewport_size = vec2(GRID_WIDTH, GRID_HEIGHT);
    vec2 position = (offsets_vertex[gl_VertexID] + cell) / viewport_size;
#elif GRID_GEOMETRY == GRID_GEOMETRY_TRIANGLE
    vec2[2][3] offsets_vertex = {
        {
            vec2(0.0, 1),
            vec2(1.0, 1),
            vec2(0.5, 0),
        },
        {
            vec2(0.5, 1),
            vec2(1.0, 0),
            vec2(0.0, 0),
        },
    };
    uint pointing_down = (cell.x + cell.y) % 2;
    vec2 offsets_instance = cell * vec2(0.5, 1.0);
    vec2 stretch = vec2(1.0, sqrt(0.75));
    vec2 viewport_size = vec2(GRID_WIDTH * 0.5 + 0.5, GRID_HEIGHT * sqrt(0.75));
    vec2 position = (offsets_vertex[pointing_down][gl_VertexID] + offsets_instance) * stretch / viewport_size;
#elif GRID_GEOMETRY == GRID_GEOMETRY_HEXAGON
    vec2[6] offsets_vertex = {
        vec2( 1.0, -0.5),
        vec2( 0.0, -1.0),
        vec2(-1.0, -0.5),
        vec2(-1.0,  0.5),
        vec2( 0.0,  1.0),
        vec2( 1.0,  0.5),
    };
    uint even_row = cell.y % 2;
    vec2 offsets_instance = vec2(1.0 + float(even_row), 1.0) + cell * vec2(2.0, 1.5);
    float cell_width_half = cos(TAU / 12.0);
    vec2 stretch = vec2(cell_width_half, 1.0);
    vec2 viewport_size = vec2((GRID_WIDTH + 0.5) * cell_width_half * 2, GRID_HEIGHT * 1.5 + 0.5);
    vec2 position = (offsets_vertex[gl_VertexID] + offsets_instance) * stretch / viewport_size;
#endif

    if (KEEP_ASPECT_RATIO) {
        // Could be done by a uniform matrix, but whatever.
        float viewport_size_aspect_ratio = viewport_size.x / viewport_size.y;
        float window_resolution_aspect_ratio = float(window_resolution.x) / float(window_resolution.y);
        float viewport_to_window_ratio = viewport_size_aspect_ratio / window_resolution_aspect_ratio;
        // the percentage of the window the viewport spans in both axes
        vec2 viewport_size_ratio_of_window;

        if (viewport_size_aspect_ratio > window_resolution_aspect_ratio) {
            // Limited by window width
            viewport_size_ratio_of_window = vec2(1.0, 1.0 / viewport_to_window_ratio);
        } else {
            // Limited by window height
            viewport_size_ratio_of_window = vec2(viewport_to_window_ratio, 1.0);
        }

        vec2 view_offset = (vec2(1.0) - viewport_size_ratio_of_window) * 0.5;

        position = view_offset + position * viewport_size_ratio_of_window;
    }

    // Transform from [0; 1]×[0; 1] to [-1; 1]×[1; -1]
    position = position * 2.0 - 1.0;
    position.y *= -1.0;
    gl_Position = vec4(position, 1.0, 1.0);

    // Debug first cell
    /* gl_Position.xyz *= float(gl_InstanceID == 0); */

    out_cell_state_current = in_cell_state_current;
    out_cell_state_previous = in_cell_state_previous;

#if CELL_STATES == 2
    vec3 color = vec3(
        float(!bool(in_cell_state_current) &&  bool(in_cell_state_previous)),
        float( bool(in_cell_state_current) && !bool(in_cell_state_previous)),
        float( bool(in_cell_state_current) &&  bool(in_cell_state_previous))
    );
#else
    vec3 color = hsv2rgb(vec3(float(in_cell_state_current - 1) / float(CELL_STATES - 1), 1.0, 1.0));
    color *= float(in_cell_state_current != 0);
#endif

    out_color = color;
}
