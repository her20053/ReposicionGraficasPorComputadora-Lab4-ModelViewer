from OpenGL.GL.shaders import * 
from OpenGL.GL import *
from math import cos
import numpy as np
from obj import * 
import pygame
import random
import glm

pygame.init()

screen = pygame.display.set_mode((800, 600),pygame.OPENGL | pygame.DOUBLEBUF)

model = Obj('./absperrhut.obj')

vertex_original = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;

uniform mat4 matrix;

out vec3 ourColor;
out vec2 fragCoord;

void main()
{
    gl_Position = matrix * vec4(position, 1.0f);
    ourColor = vertexColor;
    fragCoord = gl_Position.xy;
}
"""

fragment_original = """
#version 460
layout (location = 0) out vec4 fragColor;

uniform vec3 color;

in vec3 ourColor;

void main()
{
    //fragColor = vec4(color, 1.0f);
    fragColor = vec4(ourColor, 1.0f);
}
"""

fragment_primero = """
#version 460

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;



uniform float time;

const mat2 m = mat2( 0.80,  0.60, -0.60,  0.80 );

float noise( in vec2 p )
{
	return sin(p.x)*sin(p.y);
}

float fbm4( vec2 p )
{
    float f = 0.0;
    f += 0.5000*noise( p ); p = m*p*2.02;
    f += 0.2500*noise( p ); p = m*p*2.03;
    f += 0.1250*noise( p ); p = m*p*2.01;
    f += 0.0625*noise( p );
    return f/0.9375;
}

float fbm6( vec2 p )
{
    float f = 0.0;
    f += 0.500000*(0.5+0.5*noise( p )); p = m*p*2.02;
    f += 0.250000*(0.5+0.5*noise( p )); p = m*p*2.03;
    f += 0.125000*(0.5+0.5*noise( p )); p = m*p*2.01;
    f += 0.062500*(0.5+0.5*noise( p )); p = m*p*2.04;
    f += 0.031250*(0.5+0.5*noise( p )); p = m*p*2.01;
    f += 0.015625*(0.5+0.5*noise( p ));
    return f/0.96875;
}

vec2 fbm4_2( vec2 p )
{
    return vec2(fbm4(p), fbm4(p+vec2(7.8)));
}

vec2 fbm6_2( vec2 p )
{
    return vec2(fbm6(p+vec2(16.8)), fbm6(p+vec2(11.5)));
}

//====================================================================

float func( vec2 q, out vec4 ron )
{
    q += 0.03*sin( vec2(0.27,0.23)*time + length(q)*vec2(4.1,4.3));

	vec2 o = fbm4_2( 0.9*q );

    o += 0.04*sin( vec2(0.12,0.14)*time + length(o));

    vec2 n = fbm6_2( 3.0*o );

	ron = vec4( o, n );

    float f = 0.5 + 0.5*fbm4( 1.8*q + 6.0*n );

    return mix( f, f*f*f*3.5, f*abs(n.x) );
}

void main()
{
    vec2 iResolution = vec2(1, 1);
    vec2 p = (2.0*fragCoord-iResolution.xy)/iResolution.y;
    float e = 2.0/iResolution.y;

    vec4 on = vec4(0.0);
    float f = func(p, on);

	vec3 col = vec3(0.0);
    col = mix( vec3(0.2,0.1,0.4), vec3(0.3,0.05,0.05), f );
    col = mix( col, vec3(0.9,0.9,0.9), dot(on.zw,on.zw) );
    col = mix( col, vec3(0.4,0.3,0.3), 0.2 + 0.5*on.y*on.y );
    col = mix( col, vec3(0.0,0.2,0.4), 0.5*smoothstep(1.2,1.3,abs(on.z)+abs(on.w)) );
    col = clamp( col*f*2.0, 0.0, 1.0 );
    
#if 0
    // gpu derivatives - bad quality, but fast
	vec3 nor = normalize( vec3( dFdx(f)*iResolution.x, 6.0, dFdy(f)*iResolution.y ) );
#else    
    // manual derivatives - better quality, but slower
    vec4 kk;
 	vec3 nor = normalize( vec3( func(p+vec2(e,0.0),kk)-f, 
                                2.0*e,
                                func(p+vec2(0.0,e),kk)-f ) );
#endif    

    vec3 lig = normalize( vec3( 0.9, 0.2, -0.4 ) );
    float dif = clamp( 0.3+0.7*dot( nor, lig ), 0.0, 1.0 );
    vec3 lin = vec3(0.70,0.90,0.95)*(nor.y*0.5+0.5) + vec3(0.15,0.10,0.05)*dif;
    col *= 1.2*lin;
	col = 1.0 - col;
	col = 1.1*col*col;
    
    fragColor = vec4( col, 1.0 );
}
"""

fragment_segundo = """
#version 460

precision highp float;

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;

uniform float time;

float map(vec3 p) {
	vec3 n = vec3(0, 1, 0);
	float k1 = 1.9;
	float k2 = (sin(p.x * k1) + sin(p.z * k1)) * 0.8;
	float k3 = (sin(p.y * k1) + sin(p.z * k1)) * 0.8;
	float w1 = 4.0 - dot(abs(p), normalize(n)) + k2;
	float w2 = 4.0 - dot(abs(p), normalize(n.yzx)) + k3;
	float s1 = length(mod(p.xy + vec2(sin((p.z + p.x) * 2.0) * 0.3, cos((p.z + p.x) * 1.0) * 0.5), 2.0) - 1.0) - 0.2;
	float s2 = length(mod(0.5+p.yz + vec2(sin((p.z + p.x) * 2.0) * 0.3, cos((p.z + p.x) * 1.0) * 0.3), 2.0) - 1.0) - 0.2;
	return min(w1, min(w2, min(s1, s2)));
}

vec2 rot(vec2 p, float a) {
	return vec2(
		p.x * cos(a) - p.y * sin(a),
		p.x * sin(a) + p.y * cos(a));
}

void main() {
    vec2 iResolution = vec2(1, 1);
    float time = time;
	vec2 uv = ( fragCoord.xy / iResolution.xy ) * 2.0 - 1.0;
	uv.x *= iResolution.x /  iResolution.y;
	vec3 dir = normalize(vec3(uv, 1.0));
	dir.xz = rot(dir.xz, time * 0.23);dir = dir.yzx;
	dir.xz = rot(dir.xz, time * 0.2);dir = dir.yzx;
	vec3 pos = vec3(0, 0, time);
	vec3 col = vec3(0.0);
	float t = 0.0;
    float tt = 0.0;
	for(int i = 0 ; i < 100; i++) {
		tt = map(pos + dir * t);
		if(tt < 0.001) break;
		t += tt * 0.45;
	}
	vec3 ip = pos + dir * t;
	col = vec3(t * 0.1);
	col = sqrt(col);
	fragColor = vec4(0.05*t+abs(dir) * col + max(0.0, map(ip - 0.1) - tt), 1.0); //Thanks! Shane!
    fragColor.a = 1.0 / (t * t * t * t);
}
"""

fragment_tercero = """
#version 460
    const float SHAPE_SIZE = .618;
    const float CHROMATIC_ABBERATION = .01;
    const float ITERATIONS = 10.;
    const float INITIAL_LUMA = .5;

    const float PI = 3.14159265359;
    const float TWO_PI = 6.28318530718;

    layout (location = 0) out vec4 fragColor;
    in vec3 ourColor;
    in vec2 fragCoord;

    uniform float time;
void main(){
    vec2 iResolution = vec2(1, 1);
    vec2 uv =  (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);

    for(float i = 1.0; i < 10.0; i++){
        uv.x += 0.6 / i * cos(i * 2.5* uv.y + time);
        uv.y += 0.6 / i * cos(i * 1.5 * uv.x + time);
    }
    
    fragColor = vec4(vec3(0.1)/abs(sin(time-uv.y-uv.x)),1.0);
}
"""

compiled_vertex_original = compileShader(vertex_original, GL_VERTEX_SHADER)
compiled_fragment_original = compileShader(fragment_original, GL_FRAGMENT_SHADER)
compiled_fragment_primero = compileShader(fragment_primero, GL_FRAGMENT_SHADER)
compiled_fragment_segundo = compileShader(fragment_segundo, GL_FRAGMENT_SHADER)
compiled_fragment_tercero = compileShader(fragment_tercero, GL_FRAGMENT_SHADER)

original = compileProgram(
    compiled_vertex_original, 
    compiled_fragment_original
)


primero = compileProgram(
    compiled_vertex_original,
    compiled_fragment_primero
)

segundo = compileProgram(
    compiled_vertex_original,
    compiled_fragment_segundo
)

tercero = compileProgram(
    compiled_vertex_original,
    compiled_fragment_tercero
)

glUseProgram(primero)

vertex = []

for elem_vertice in range(len(model.vertices)):
    for elem in range(len(model.vertices[elem_vertice])):
        vertex.append(model.vertices[elem_vertice][elem])

vertex_data = np.array(vertex, dtype=np.float32)



vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(
    GL_ARRAY_BUFFER,
    vertex_data.nbytes, 
    vertex_data, 
    GL_STATIC_DRAW
)

glVertexAttribPointer(
    0, 
    3,
    GL_FLOAT,
    GL_FALSE,
    3 * 4,
    ctypes.c_void_p(0)
)

glEnableVertexAttribArray(0)

faces = []
for elem_face in range(len(model.faces)):
    for elem in range(len(model.faces[elem_face])):
        faces.append(int(model.faces[elem_face][elem][0])-1)

faces_data = np.array(faces, dtype=np.int32)


element_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces_data.nbytes, faces_data, GL_STATIC_DRAW)



def calculateMatrix(angle, arrRot, translate):

    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(0, 0, 0))
    rotate = glm.rotate(i, glm.radians(angle), arrRot)
    scale = glm.scale(i, glm.vec3(1.25, 1.25, 1.25))

    model = translate * rotate * scale

    view = glm.lookAt(
        glm.vec3(0, 0, 5),
        glm.vec3(0, 0, 0),
        glm.vec3(0, 1, 0)
    )

    projection = glm.perspective(
        glm.radians(45),
        1600 / 1200,
        0.1,
        1000
    )
    glViewport(-870, -1100, 2500, 2500)

    matrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(original, "matrix"),
        1,
        GL_FALSE,
        glm.value_ptr(matrix)
    )

running = True

glClearColor(0.0, 0.0, 0.0, 1.0)
r = 0
l = 0
setshdr = False
indice = 0
efectos = [primero, segundo, tercero, original]
currentoriginal = primero
arrRot = glm.vec3(0, 1, 0)

prev_time = pygame.time.get_ticks()

while running:
    glClear(GL_COLOR_BUFFER_BIT)

    if setshdr:
        glUseProgram(efectos[indice])
        currentoriginal = efectos[indice]
        setshdr = False
    
    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1, color2, color3)
    glUniform3fv(
        glGetUniformLocation(currentoriginal, "color"),
        1,
        glm.value_ptr(color)
    )

    time = (pygame.time.get_ticks() - prev_time) / 1000
    glUniform1f(
        glGetUniformLocation(currentoriginal, "time"),
        time
    )

    glDrawElements(GL_TRIANGLES, len(faces_data), GL_UNSIGNED_INT, None)

    calculateMatrix(r, arrRot, 0)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if r > 0 and r < 2000:
        indice = 0
        setshdr = True
    if r > 2000 and r < 3000:
        indice = 1
        setshdr = True
    if r > 3000 and r < 5000:
        indice = 2
        setshdr = True
    if r > 5000:
        r = 0
    
    keys = pygame.key.get_pressed()
    arrRot = glm.vec3(90, l/10, l/50)
    r+=1
    l+=1
            
