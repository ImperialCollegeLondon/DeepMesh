###############################################################################################################
## To align subject meshes with an atlas and transform all the subjects from subject-space to atlas-space.
## By Soodeh
## Sep 2024
###############################################################################################################

import os
import subprocess
from tqdm import tqdm  # Progress bar for iteration
import numpy as np
import nibabel as nib
import vtk
import tempfile  # For handling temporary files
import time
import shutil    # For removing directories
import argparse  # For command-line argument parsing

# Function to read a VTK file using VTK library
def read_vtk(file_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()  # Update the reader to read the file
    return reader.GetOutput()

# Function to write a VTK file using VTK library
def write_vtk(file_path, polydata):
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToASCII()  # Save in ASCII format, or SetFileTypeToBinary() for binary
    writer.SetFileVersion(42)  # Set VTK file version 5.1
    writer.Write()

# Function to check if there is scaling in the affine matrix
def check_for_scaling(affine_matrix):
    # Extract the 3x3 part of the matrix (rotation + scaling)
    rotation_scaling_matrix = affine_matrix[:3, :3]
    
    # Compute the norms of the columns (should be 1 for no scaling)
    column_norms = np.linalg.norm(rotation_scaling_matrix, axis=0)
    
    # Check if the column norms are not equal to 1 (indicating scaling)
    if np.allclose(column_norms, [1, 1, 1]):
        print("No scaling is present in the affine matrix.")
    else:
        print(f"Scaling is present in the affine matrix. Column norms: {column_norms}")

# Function to check if there is shearing in the affine matrix
def check_for_shearing(affine_matrix):
    # Extract the 3x3 submatrix (rotation, scaling, and shearing)
    rotation_scaling_matrix = affine_matrix[:3, :3]
    
    # Check for shearing: off-diagonal elements should be zero for no shearing
    shear_elements = rotation_scaling_matrix - np.diag(np.diag(rotation_scaling_matrix))
    
    # If any off-diagonal elements are non-zero, shearing is present
    if np.any(np.abs(shear_elements) > 1e-6):  # Tolerance for floating point numbers
        print("Shearing is present in the affine matrix.")
    else:
        print("No shearing is present in the affine matrix.")

# Function to remove scaling from an affine matrix
def remove_scaling_from_affine(affine_matrix):
    # Extract the 3x3 part of the matrix (rotation + scaling)
    rotation_matrix = affine_matrix[:3, :3]
    
    # Normalize each column to remove scaling
    rotation_matrix_normalized = rotation_matrix / np.linalg.norm(rotation_matrix, axis=0)
    
    # Create a new affine matrix with normalized rotation and original translation
    affine_no_scaling = np.eye(4)
    affine_no_scaling[:3, :3] = rotation_matrix_normalized
    affine_no_scaling[:3, 3] = affine_matrix[:3, 3]  # Keep the translation part
    
    return affine_no_scaling

# Main processing function for subjects
def process_subjects(UKBB_DIR, MESH_DIR, ATLAS_DIR, OUTPUT_DIR, SECTION, N_FRAME):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = os.listdir(MESH_DIR)
    print('data:', len(data))

    # Load the atlas affine and atlas mesh
    atlas_imdir = f'{ATLAS_DIR}/template.nii.gz'
    atlas_img = nib.load(atlas_imdir)
    affine_atlas = atlas_img.affine

    # Use VTK to read the atlas mesh
    atlas_path = f'{ATLAS_DIR}/myo_ED.vtk'
    atlas_mesh = read_vtk(atlas_path)

    # Extract atlas faces (if needed for later comparisons)
    atlas_faces = atlas_mesh.GetPolys()

    for subject in tqdm(data, desc='Processing', unit='subject'):
        print('ID', subject)

        start_time = time.time()

        folder_align = f'{OUTPUT_DIR}/{subject}/{SECTION}_template_space'

        # Check if the folder exists
        if os.path.isdir(folder_align):
            # Check if the folder contains files
            if os.listdir(folder_align):  # This will return a list of files/folders
                print(f"{folder_align} exists and contains files.")
            else:
                print(f"{folder_align} exists but is empty.")
        else:
            print(f"{folder_align} does not exist. Creating it.")
            os.makedirs(folder_align, exist_ok=True)

            sub_dir = f'{UKBB_DIR}/{subject}/4D_rview/4Dimg.nii.gz'
            subject_img = nib.load(sub_dir)
            affine_subject = subject_img.affine

            # check_for_scaling(affine_subject)
            # check_for_shearing(affine_subject)

            # Remove scaling from both affine matrices (only rotation and translation)
            affine_subject_no_scaling = remove_scaling_from_affine(affine_subject)
            affine_atlas_no_scaling = remove_scaling_from_affine(affine_atlas)

            # Compute the transformation matrix (without scaling)
            transformation_matrix = np.dot(affine_atlas_no_scaling, np.linalg.inv(affine_subject_no_scaling))

            # Iterate over frames
            for frame in range(N_FRAME):
                moving_mesh_fr = f'{MESH_DIR}/{subject}/vtkfile/mesh_{frame:02d}.vtk'

                # Read the subject's mesh for the given frame using VTK
                moving_mesh = read_vtk(moving_mesh_fr)

                # Get the points from the mesh
                points = np.array([moving_mesh.GetPoint(i) for i in range(moving_mesh.GetNumberOfPoints())])

                # Apply the affine transformation (without scaling)
                transformed_points = nib.affines.apply_affine(transformation_matrix, points)

                # Optionally apply flip along axes (depending on visual results)
                transformed_points[:, 2] = -transformed_points[:, 2] ## Z-axis
                transformed_points[:, 1] = -transformed_points[:, 1] ## Y-axis

                # Create a new vtkPoints object and set the transformed points
                vtk_points = vtk.vtkPoints()
                for p in transformed_points:
                    vtk_points.InsertNextPoint(p)

                # Create a new vtkPolyData object to store the transformed mesh
                transformed_mesh = vtk.vtkPolyData()
                transformed_mesh.SetPoints(vtk_points)
                transformed_mesh.SetPolys(atlas_mesh.GetPolys())  # Set the faces from the atlas
                
                # Write the transformed mesh to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.vtk') as temp_vtk_file:
                    transformed_mesh_path = temp_vtk_file.name

                    write_vtk(transformed_mesh_path, transformed_mesh)

                    dof_folder = f'{OUTPUT_DIR}/{subject}/{SECTION}_dofs'
                    os.makedirs(dof_folder, exist_ok=True)
                    dofout_file_rreg = f'{dof_folder}/dof_rreg.dof.gz'

                    # Run srreg on the temporary file (only for the first frame)
                    if frame == 0:
                        srreg_cmd = (
                            f"srreg {transformed_mesh_path} {atlas_path} "
                            f"-dofout {dofout_file_rreg} "
                            f"-symmetric "
                        )
                        subprocess.run(srreg_cmd, shell=True, check=True)

                    # Run ptransformation to apply the transformation
                    aligned_mesh_path = f'{folder_align}/{SECTION}_fr{frame:02d}.vtk'

                    ptransformation_cmd = (
                        f"ptransformation {transformed_mesh_path} {aligned_mesh_path} "
                        f"-dofin {dofout_file_rreg}"
                    )
                    subprocess.run(ptransformation_cmd, shell=True, check=True)
            
            # Remove the temporary files after use
            if os.path.exists(dof_folder):
                shutil.rmtree(dof_folder)  # Remove the entire directory and its contents
        

        print(f"Processed and aligned all frames for subject {subject}")
        print("--- %s seconds ---" % (time.time() - start_time))

        # # Remove the temporary files after use
        # if os.path.exists(dof_folder):
        #     shutil.rmtree(dof_folder)  # Remove the entire directory and its contents

    print("Processing complete.")

# Define main function to handle argparse and run process
def main():
    parser = argparse.ArgumentParser(description="Process 4D cardiac data using affine transformations.")
    parser.add_argument('--UKBB_DIR', type=str, required=True, help='Directory for UKBB data')
    parser.add_argument('--MESH_DIR', type=str, required=True, help='Directory for subject mesh data')
    parser.add_argument('--ATLAS_DIR', type=str, required=True, help='Directory for atlas data')
    parser.add_argument('--OUTPUT_DIR', type=str, required=True, help='Directory for output data')
    parser.add_argument('--SECTION', type=str, default='LVmyo', help='Section to process')
    parser.add_argument('--N_FRAME', type=int, default=50, help='Number of frames to process')

    args = parser.parse_args()

    process_subjects(
        args.UKBB_DIR,
        args.MESH_DIR,
        args.ATLAS_DIR,
        args.OUTPUT_DIR,
        args.SECTION,
        args.N_FRAME
    )

if __name__ == "__main__":
    main()
