import numpy as np
from stl import mesh


def scale_stl(input_file, output_file, scale_factor):
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(input_file)

    # Apply the scaling transformation
    your_mesh.vectors *= scale_factor

    # Save the scaled STL file
    your_mesh.save(output_file)
    print(f"Scaled STL file saved as: {output_file}")


# Parameters
input_file = 'Path to Project directory/models/new origin/F1Car.stl'  # Replace with your input STL file path
output_file = 'Path to Project directory/models/STL/F1Car30.stl'  # Replace with your desired output STL file path
scale_factor = 30  # Replace with your desired scale factor

# Run the scaling function
scale_stl(input_file, output_file, scale_factor)
