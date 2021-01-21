import numpy as np

_image_vert_tex = np.array((
    (-1, -1, 0, 1),
    (1, -1, 1, 1),
    (1, 1, 1, 0),
    (-1, 1, 0, 0)
))[np.array([(0, 1, 2), (2, 3, 0)])].astype('f4').reshape(-1)

_image_vertex_shader = """
    #version 330
    uniform mat4 mvp;
    in vec2 vert;
    in vec2 tex_coord;
    out vec2 v_tex_coord;
    void main(){
        gl_Position = mvp * vec4(vert, 0., 1.);
        v_tex_coord = tex_coord;
    }
"""

_image_fragment_shader = """
    #version 330
    uniform sampler2D tex;
    in vec2 v_tex_coord;
    out vec4 color;
    void main(){
        color = vec4(texture(tex, v_tex_coord).rgb, 1.);
        //color = vec4(1, 0, 0, 1);
    }
"""

_line_vertex_shader = """
    #version 330
    uniform vec4 color;
    in vec2 in_vert;
    out vec4 v_color;
    void main(){
        vec2 vert = in_vert;
        gl_Position = vec4(vert * 2 - 1., 0., 1.);
        v_color = color;
    }
"""

_line_fragment_shader = """
    #version 330
    in vec4 v_color;
    out vec4 f_color;
    void main(){
        f_color = v_color;
    }
"""

_model_vertex_shader = """
    #version 330
    uniform mat4 mvp;
    uniform mat4 mv;
    uniform float alpha;
    uniform vec3 bb0;
    uniform vec3 bb1;
    in vec3 in_vert;
    in vec3 in_norm;
    out vec3 v_vert_obj_norm;
    out vec3 v_vert_cam;
    out vec3 v_norm;
    out float v_alpha;
    void main() {
        gl_Position = mvp * vec4(in_vert, 1.0);
        v_vert_cam = (mv * vec4(in_vert, 1.0)).xyz;
        v_vert_obj_norm = (in_vert - bb0) / (bb1 - bb0);
        v_norm = mat3(mv) * in_norm;
        v_alpha = alpha;
    }
"""

_model_fragment_shader = """
    #version 330
    in vec3 v_vert_cam;
    in vec3 v_vert_obj_norm;
    in vec3 v_norm;
    in float v_alpha;
    out vec4 f_color;
    void main() {
        vec3 norm = normalize(v_norm);
        float a = -dot(norm, normalize(v_vert_cam));
        vec3 color = (0.2 + 0.8 * a) * v_vert_obj_norm;
        f_color = f_color = vec4(color, v_alpha);
    }
"""

_point_vertex_shader = """
    #version 330
    uniform mat4 mvp;
    uniform mat3 rot;
    uniform vec3 pos;
    uniform vec4 color;
    uniform float scale;
    
    in vec3 in_vert;
    in vec3 in_norm;
    out vec3 v_norm;
    out vec4 v_color;
    void main() {
        gl_Position = mvp * vec4(in_vert * scale + pos, 1.0);
        v_norm = rot * in_norm;
        v_color = color;
    }
"""

_point_fragment_shader = """
    #version 330
    in vec3 v_norm;
    in vec4 v_color;
    out vec4 f_color;
    void main() {
        vec3 norm = normalize(v_norm);
        vec3 color = vec3(v_color);
        color = (0.4 + 0.6 * dot(norm, vec3(0, 0, 1))) * color;
        f_color = vec4(color, v_color.a);
    }
"""
