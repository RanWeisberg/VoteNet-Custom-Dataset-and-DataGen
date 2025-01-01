import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
import trimesh

def transform_coordinate_system(vertices):
    """
    Apply transformation to switch the coordinate system to:
    Z is up, Y is forward, X is right-ward.
    """
    transformation_matrix = np.array([[0, 1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]])
    return np.dot(vertices, transformation_matrix.T)

def render_to_depth_buffer(stl_path, heading_angle, image_size=(245, 192), distance=10.0):
    # Initialize Pygame
    pygame.init()
    pygame.display.set_mode(image_size, pygame.OPENGL) # | pygame.DOUBLEBUF)
    pygame.display.set_caption("Depth Buffer Rendering")

    # Load the STL file
    mesh = trimesh.load_mesh(stl_path)
    mesh.apply_scale(0.001)

    # Transform the vertices to the new coordinate system
    mesh.vertices = transform_coordinate_system(mesh.vertices)

    # Compute the camera position based on the given heading angle
    heading = heading_angle + np.pi

    # Set the camera position around the origin based on heading and distance
    camera_x = distance * np.sin(heading)
    camera_y = distance * np.cos(heading)
    camera_z = 0  # No change in the z-axis

    # Camera up vector remains constant
    up_x = 0
    up_y = 0
    up_z = 1

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, image_size[0] / image_size[1], 0.1, 1000)

    # Set up the modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(camera_x, camera_y, camera_z,  # Camera position
              0, 0, 0,                      # Look at the origin
              up_x, up_y, up_z)             # Up vector

    # Enable depth test
    glEnable(GL_DEPTH_TEST)

    # Render to the depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    for face in mesh.faces:
        for vertex_index in face:
            vertex = mesh.vertices[vertex_index]
            glVertex3f(vertex[0], vertex[1], vertex[2])
    glEnd()
    glFlush()
    #pygame.display.flip()

    # Read depth buffer
    depth_buffer = glReadPixels(0, 0, image_size[0], image_size[1], GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_buffer = np.frombuffer(depth_buffer, dtype=np.float32).reshape(image_size[1], image_size[0])

    # Retrieve matrices and viewport
    modelview_matrix = np.array(glGetDoublev(GL_MODELVIEW_MATRIX)).reshape(4, 4)
    projection_matrix = np.array(glGetDoublev(GL_PROJECTION_MATRIX)).reshape(4, 4)
    viewport = np.array(glGetIntegerv(GL_VIEWPORT))

    # Quit Pygame
    pygame.quit()

    # Convert depth buffer to point cloud
    point_cloud = []
    for y in range(image_size[1]):
        for x in range(image_size[0]):
            z = depth_buffer[y, x]
            if z < 1.0 and not np.isnan(z) and not np.isinf(z):  # Valid depth values
                # Get window coordinates
                winX = x
                winY = y #image_size[1] - y  - 1
                winZ = z

                # Unproject the window coordinates to get the object coordinates
                obj_coords = gluUnProject(winX, winY, winZ, modelview_matrix, projection_matrix, viewport)
                point_cloud.append(obj_coords)


    return np.array(point_cloud)

# Example usage
if __name__ == "__main__":
    stl_path = "example.stl"
    heading_angle = 45  # example heading angle
    point_cloud = render_to_depth_buffer(stl_path, heading_angle)
    print("Generated point cloud:", point_cloud)
