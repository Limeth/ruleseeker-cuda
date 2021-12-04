// (1) Prepended version directive
// (2) Prepended common.h header
layout (location = 0) in flat uint in_cell_state_current;
layout (location = 1) in flat uint in_cell_state_previous;
layout (location = 2) in flat vec3 in_color;
layout (location = 3) in flat uint in_cell_index;

layout (location = 0) out vec4 out_color;
layout (location = 1) out uint out_cell_index;

void main() {
    out_color = vec4(in_color, 1.0);
    out_cell_index = in_cell_index;
} 
