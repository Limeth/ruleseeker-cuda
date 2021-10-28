#pragma once
#include <GL/glut.h>

// struktura bitmapy
typedef struct {
    int width;          // sirka bitmapy v pixelech
    int height;         // vyska bitmapy v pixelech
    uchar4 *pixels;     // ukazatel na bitmapu na strane CPU
    uchar4 *deviceData; // ukazatel na data bitmapy na GPU
} bitmap_t;

bitmap_t *bitmap = NULL;

// vykresleni bitmapy v OpenGL
__host__ void draw_image() {
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );

    if(bitmap != NULL) {
        if(bitmap->pixels != NULL)
            glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
    }

    glutSwapBuffers();
}


int ui_loop(
        int argc,
        char **argv,
        int width,
        int height,
        void (handle_keys)(unsigned char key, int x, int y),
        void (idle_func)()
) {
    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutCreateWindow("Life Game");
    glutDisplayFunc(draw_image);
    glutKeyboardFunc(handle_keys);
    glutIdleFunc(idle_func);
    glutMainLoop();

    return 0;
}
