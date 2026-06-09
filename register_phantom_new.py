import os
import copy
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

# --- USER CONFIGURATION PARAMETERS ---
MODEL_PCD_PATH = r"C:\Users\Admin\Documents\phantom_registration\dense_phantom_perfect.pcd"  #source point cloud
MARKER_SIZE = 0.05     
TARGET_ID = 0           # ArUco Tag ID
NUM_SAMPLES_NEEDED = 10 # Number of iterations to average
# --------------------------------------

def get_realsense_intrinsics(profile):
    """Extracts factory calibration configurations directly from the stream hardware."""
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    K = np.array([
        [intrinsics.fx, 0,             intrinsics.ppx],
        [0,             intrinsics.fy, intrinsics.ppy],
        [0,             0,             1             ]
    ], dtype=np.float32)
    
    D = np.zeros(5, dtype=np.float32)
    return K, D, intrinsics

def execute_icp_alignment(source_model, target_scene, initial_guess):
    """High-accuracy Multi-Scale Point-to-Plane ICP pipeline."""
    source_model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    target_scene.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    voxel_radii = [0.005, 0.002, 0.001]
    max_iter_per_scale = [50, 30, 15]
    
    current_transformation = initial_guess
    icp_result = None
    
    for i, voxel_size in enumerate(voxel_radii):
        distance_threshold = voxel_size * 2.5
        source_down = source_model.voxel_down_sample(voxel_size)
        target_down = target_scene.voxel_down_sample(voxel_size)
        
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter_per_scale[i]
        )
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria
        )
        current_transformation = icp_result.transformation
        
    return current_transformation, icp_result.fitness

def compute_average_transformation(matrix_list):
    """Computes a mathematically valid average of multiple 4x4 homogeneous matrices."""
    translations = []
    quaternions = []
    
    for T in matrix_list:
        translations.append(T[:3, 3])
        # Extract rotation and convert to quaternion representation
        rotation_matrix = T[:3, :3]
        quat = R.from_matrix(rotation_matrix).as_quat()
        quaternions.append(quat)
        
    # 1. Compute basic arithmetic mean for linear translations
    avg_translation = np.mean(translations, axis=0)
    
    # 2. Compute true spherical orientation mean using Scipy's optimized structural averaging
    avg_rotation_obj = R.mean(R.from_quat(quaternions))
    avg_rotation_matrix = avg_rotation_obj.as_matrix()
    
    # 3. Reconstruct into a unified 4x4 homogeneous matrix
    avg_T = np.eye(4)
    avg_T[:3, :3] = avg_rotation_matrix
    avg_T[:3, 3] = avg_translation
    return avg_T

def main():
    if not os.path.exists(MODEL_PCD_PATH):
        raise FileNotFoundError(f"Missing dense reference point cloud at: {MODEL_PCD_PATH}")

    print("[*] Loading reference model point cloud...")
    ref_model = o3d.io.read_point_cloud(MODEL_PCD_PATH)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    camera_matrix, dist_coeffs, intrinsic_obj = get_realsense_intrinsics(profile)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    half_s = MARKER_SIZE / 2.0
    marker_obj_pts = np.array([
        [-half_s, half_s, 0], [half_s, half_s, 0], 
        [half_s, -half_s, 0], [-half_s, -half_s, 0]
    ], dtype=np.float32)

    # List container to store individual scan iterations
    collected_matrices = []

    print("\n==================================================================")
    print(f" PIPELINE ACTIVE: Collect {NUM_SAMPLES_NEEDED} samples via [SPACEBAR].")
    print("==================================================================\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data()) 
            
            display_img = color_img.copy()
            corners, ids, _ = detector.detectMarkers(color_img)
            marker_valid = False
            
            if ids is not None and TARGET_ID in ids.flatten():
                marker_valid = True
                cv2.aruco.drawDetectedMarkers(display_img, corners, ids, borderColor=(0, 255, 0))
                status_text = f"READY: {len(collected_matrices)}/{NUM_SAMPLES_NEEDED} Samples - Press SPACE"
                cv2.putText(display_img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                status_text = f"SEARCHING... ({len(collected_matrices)}/{NUM_SAMPLES_NEEDED} Collected)"
                cv2.putText(display_img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("RealSense Live View", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == 32:  # Spacebar
                if not marker_valid:
                    print("[!] Target ArUco marker is not visible.")
                    continue
                
                idx = np.where(ids.flatten() == TARGET_ID)[0][0]
                success, rvec, tvec = cv2.solvePnP(marker_obj_pts, corners[idx][0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    print("[!] Error solving PnP geometry.")
                    continue

                R_cam_marker, _ = cv2.Rodrigues(rvec)
                Cam_T_Marker = np.eye(4)
                Cam_T_Marker[:3, :3] = R_cam_marker
                Cam_T_Marker[:3, 3] = tvec.squeeze()
                
                o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    intrinsic_obj.width, intrinsic_obj.height, intrinsic_obj.fx, intrinsic_obj.fy, intrinsic_obj.ppx, intrinsic_obj.ppy
                )
                o3d_depth = o3d.geometry.Image(depth_data)
                o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
                
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=1.2, convert_rgb_to_intensity=False
                )
                scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsic)
                
                marker_center = Cam_T_Marker[:3, 3]
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=marker_center - np.array([0.3, 0.3, 0.3]),
                    max_bound=marker_center + np.array([0.3, 0.3, 0.05])
                )
                cropped_scene = scene_pcd.crop(bbox)
                
                if len(cropped_scene.points) < 10:
                    print("[!] Crop Error: No coordinates found within bounding box.")
                    continue

                scene_center = cropped_scene.get_center()
                initial_guess_center = np.eye(4)
                initial_guess_center[:3, 3] = scene_center
                initial_guess_center[:3, :3] = np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ], dtype=np.float32)
                
                Model_T_Cam, fitness = execute_icp_alignment(ref_model, cropped_scene, initial_guess_center)
                
                Cam_T_Model = Model_T_Cam
                Marker_T_Model = np.linalg.inv(Cam_T_Marker) @ Cam_T_Model
                
                # Append the calculated matrix into our array matrix pool
                collected_matrices.append(Marker_T_Model)
                print(f"[+] Sample {len(collected_matrices)}/{NUM_SAMPLES_NEEDED} saved successfully. (ICP Fitness: {fitness:.4f})")
                
                # Check if we have completed our batch collection goal
                if len(collected_matrices) == NUM_SAMPLES_NEEDED:
                    print("\n[*] Processing batch calculation averages across all coordinates...")
                    final_averaged_matrix = compute_average_transformation(collected_matrices)
                    
                    print("\n====================================================================")
                    print(f" FINAL COMBINED GEOMETRIC AVERAGE MATRIX ({NUM_SAMPLES_NEEDED} Scans):")
                    print("====================================================================")
                    for row in final_averaged_matrix:
                        print(f"  [ {row[0]:.4f},  {row[1]:.4f},  {row[2]:.4f}  |  {row[3]:.4f} ]")
                    print("====================================================================\n")
                    
                    # --- ADDED: VISUALIZE THE FINAL AVERAGED RESULT ---
                    print("[*] Launching Final Averaged Verification Window...")
                    print("    - Red Shape: Your CAD model transformed by the AVERAGED matrix.")
                    print("    - Colored Points: The actual physical scan.")
                    print("    - CLOSE THIS WINDOW to reset and start a new batch.")
                    
                    # 1. Create a deep copy of the reference CAD model
                    averaged_model = copy.deepcopy(ref_model)
                    
                    # 2. Transform the CAD model using the final averaged Marker_T_Model matrix
                    # Note: Because draw_geometries visualizes in the Camera frame, 
                    # we need to transform the model back to the camera space using the latest Cam_T_Marker link.
                    Cam_T_Model_Avg = Cam_T_Marker @ final_averaged_matrix
                    averaged_model.transform(Cam_T_Model_Avg)
                    
                    # 3. Paint the CAD model a distinct color (e.g., solid red) to easily see alignment contrast
                    averaged_model.paint_uniform_color([1.0, 0.0, 0.0]) 
                    
                    # 4. Pop up the window containing the last captured scene and your averaged CAD model
                    o3d.visualization.draw_geometries(
                        [cropped_scene, averaged_model], 
                        window_name=f"FINAL AVERAGE VERIFICATION ({NUM_SAMPLES_NEEDED} SCANS)"
                    )
                    # --------------------------------------------------
                    
                    # Wipe the cache list clean so you can take a fresh batch of 10 if desired
                    collected_matrices = []
                    print("System reset. Ready to record a new batch of 10 samples.\n")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()