// (1) Prepended version directive
// (2) Prepended config.h header
layout (location = 0) in flat uint in_cell_state_current;
layout (location = 1) in flat uint in_cell_state_previous;
layout (location = 2) in flat vec3 in_color;

out vec4 out_color;

void main() {
    out_color = vec4(in_color, 1.0);
} 
