// (1) Prepended version directive
// (2) Prepended config.h header
layout (location = 0) in flat uint in_cell_state_current;
layout (location = 1) in flat uint in_cell_state_previous;

out vec4 out_color;

void main() {
#if CELL_STATES == 2
    vec3 color = vec3(
        float(!bool(in_cell_state_current) &&  bool(in_cell_state_previous)),
        float( bool(in_cell_state_current) && !bool(in_cell_state_previous)),
        float( bool(in_cell_state_current) &&  bool(in_cell_state_previous))
    );
#endif

    out_color = vec4(color, 1.0);
} 
