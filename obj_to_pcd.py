import open3d as o3d
import os
import sys

def convert_obj_to_dense_pcd(input_obj_path, output_pcd_path, num_points=500000):
    """
    Loads an OBJ file, samples it to create a dense point cloud, and saves it as a PCD file.
    
    Parameters:
    -----------
    input_obj_path : str
        Path to the incoming .obj file.
    output_pcd_path : str
        Path where the final .pcd file should be written.
    num_points : int
        The total number of points to sample across the mesh surfaces. 
        Higher numbers mean a denser point cloud. Default is 500k points.
    """
    if not os.path.exists(input_obj_path):
        print(f"Error: Input file '{input_obj_path}' does not exist.")
        return False

    print(f"[*] Loading 3D mesh: {input_obj_path}")
    # Load the triangle mesh model
    mesh = o3d.io.read_triangle_mesh(input_obj_path)
    
    if mesh.is_empty():
        print("Error: Could not load the mesh. The file may be corrupt or empty.")
        return False

    print(f"[*] Successfully loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    print(f"[*] Sampling mesh surface into a dense cloud of {num_points:,} points...")
    
    # Use Poisson Disk Sampling for a highly uniform, dense distribution of points
    # Fallback to sample_points_uniformly if the mesh has broken geometry/non-manifold edges
    try:
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    except Exception as e:
        print(f"[!] Poisson sampling failed ({e}). Falling back to uniform triangle sampling...")
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    print(f"[*] Saving dense point cloud to: {output_pcd_path}")
    # Save the point cloud data
    success = o3d.io.write_point_cloud(output_pcd_path, pcd)
    
    if success:
        print("[+] Conversion completed successfully!")
        return True
    else:
        print("[-] Error: Failed to write the PCD file.")
        return False

if __name__ == "__main__":
    # Example Configuration
    # Replace these with your real path files
    INPUT_OBJ = r"C:\Users\Admin\Documents\phantom_registration\Oval_ring_perfect.obj"
    OUTPUT_PCD = r"dense_phantom_perfect_2k.pcd"
    
    # Change this number to control density (e.g., 100,000 for low, 1,000,000 for extreme density)
    TARGET_DENSITY = 2500 
    
    convert_obj_to_dense_pcd(INPUT_OBJ, OUTPUT_PCD, TARGET_DENSITY)