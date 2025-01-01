import numpy as np
from stl import mesh


def translate_stl(input_file, output_file, translation_vector):
    """
    Translates the origin of an STL file by a specified number of millimeters in the X, Y, or Z direction.

    Parameters:
        input_file (str): Path to the input STL file.
        output_file (str): Path to save the translated STL file.
        translation_vector (tuple or list): A 3-element vector specifying the translation in X, Y, and Z directions.
                                            For example, (10, 0, 0) will move the mesh 10 mm along the X-axis.
    """
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(input_file)

    # Apply the translation transformation
    your_mesh.translate(translation_vector)

    # Save the translated STL file
    your_mesh.save(output_file)
    print(f"Translated STL file saved as: {output_file}")


# Example usage
input_file = 'Path to Project directory/models/new origin/F1Car.stl'  # Replace with your input STL file path
output_file = 'Path to Project directory/models/new origin/F1Car.stl'  # Replace with your desired output STL file path

# Specify the translation vector in millimeters (move 10 mm along the X-axis, 20 mm along the Y-axis, and 0 mm along the Z-axis)
translation_vector = (-10, -10, 0)

# Run the translation function
translate_stl(input_file, output_file, translation_vector)
