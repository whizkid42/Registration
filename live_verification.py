import os
import copy
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# --- USER CONFIGURATION PARAMETERS ---
MODEL_PCD_PATH = r"C:\Users\Admin\Documents\phantom_registration\dense_phantom_perfect.pcd"
MARKER_SIZE = 0.05     
TARGET_ID = 0           # ArUco Tag ID
VOXEL_DOWN_SIZE = 0.003 # 3mm point density for clean coverage
# --------------------------------------

# --- YOUR INJECTED MASTER CALIBRATION MATRIX ---
FINAL_AVERAGED_MATRIX = np.array([
    [0.9982, -0.0359, -0.0477,  0.1487],
    [0.0318,  0.9959, -0.0845, -0.0031],
    [0.0506,  0.0828,  0.9953,  0.0360],
    [0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)
# -----------------------------------------------

def get_realsense_intrinsics(profile):
    """Extracts factory calibration configurations directly from the stream hardware."""
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    K = np.array([
        [intrinsics.fx, 0,             intrinsics.ppx],
        [0,             intrinsics.fy, intrinsics.ppy],
        [0,             0,             1             ]
    ], dtype=np.float32)
    
    D = np.array(intrinsics.coeffs, dtype=np.float32)
    return K, D, intrinsics

def main():
    if not os.path.exists(MODEL_PCD_PATH):
        raise FileNotFoundError(f"Missing dense reference point cloud at: {MODEL_PCD_PATH}")

    print("[*] Loading reference model point cloud asset...")
    ref_model = o3d.io.read_point_cloud(MODEL_PCD_PATH)
    
    print(f"[*] Optimizing cloud density for video overlay (Voxel: {VOXEL_DOWN_SIZE*1000}mm)...")
    ref_model_down = ref_model.voxel_down_sample(VOXEL_DOWN_SIZE)
    
    # Extract the raw Nx3 matrix array of 3D coordinates from Open3D
    cad_points_3d = np.asarray(ref_model_down.points)
    
    # Convert points to homogeneous coordinates (Nx4) by appending a column of 1s
    cad_points_homog = np.hstack((cad_points_3d, np.ones((cad_points_3d.shape[0], 1)))).T

    # Configure RealSense Hardware Pipeline Environment (UPDATED TO 640x480 @ 15 FPS)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    camera_matrix, dist_coeffs, _ = get_realsense_intrinsics(profile)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    half_s = MARKER_SIZE / 2.0
    marker_obj_pts = np.array([
        [-half_s, half_s, 0], [half_s, half_s, 0], 
        [half_s, -half_s, 0], [-half_s, -half_s, 0]
    ], dtype=np.float32)

    print("\n==================================================================")
    print(" PIPELINE ACTIVE: 640x480 @ 15 FPS configuration initialized.")
    print(" Point camera at ArUco tag to see the 3D overlay.")
    print(" Press [q] inside the frame window to terminate.")
    print("==================================================================\n")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            display_img = color_img.copy()
            
            corners, ids, _ = detector.detectMarkers(color_img)
            
            if ids is not None and TARGET_ID in ids.flatten():
                idx = np.where(ids.flatten() == TARGET_ID)[0][0]
                success, rvec, tvec = cv2.solvePnP(marker_obj_pts, corners[idx][0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                
                if success:
                    # Draw ArUco boundary lines for reference tracking visual stability
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids, borderColor=(0, 255, 0))
                    
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    Cam_T_Marker = np.eye(4)
                    Cam_T_Marker[:3, :3] = R_cam_marker
                    Cam_T_Marker[:3, 3] = tvec.squeeze()
                    
                    # 1. Compute the full dynamic transformation path for this frame
                    Cam_T_Model = Cam_T_Marker @ FINAL_AVERAGED_MATRIX
                    
                    # 2. Multiply all CAD model coordinates simultaneously to jump to Camera Frame
                    points_in_cam = Cam_T_Model @ cad_points_homog  # Result is a 4xN matrix
                    
                    # 3. Separate structural parameters for mathematical filtering
                    X_c = points_in_cam[0, :]
                    Y_c = points_in_cam[1, :]
                    Z_c = points_in_cam[2, :]
                    
                    # 4. Filter vectors: Only project points that are physically in front of the lens
                    front_indices = np.where(Z_c > 0.01)[0]
                    
                    if len(front_indices) > 0:
                        # Reassemble active coordinates (3xN)
                        pts_to_project = np.vstack((X_c[front_indices], Y_c[front_indices], Z_c[front_indices]))
                        
                        # 5. Fast perspective projection mapping onto 2D image matrix space
                        pixel_homog = camera_matrix @ pts_to_project
                        u_arr = (pixel_homog[0, :] / pixel_homog[2, :]).astype(np.int32)
                        v_arr = (pixel_homog[1, :] / pixel_homog[2, :]).astype(np.int32)
                        
                        # 6. Bounds check: Filter points that reside inside our active 640x480 pixel dimensions
                        valid_mask = (u_arr >= 0) & (u_arr < display_img.shape[1]) & \
                                     (v_arr >= 0) & (v_arr < display_img.shape[0])
                        
                        u_final = u_arr[valid_mask]
                        v_final = v_arr[valid_mask]
                        
                        # 7. Draw the 3D model footprint point array over the live frames (Cyan highlight color)
                        for u, v in zip(u_final, v_final):
                            display_img[v, u] = [255, 255, 0]  # Cyan coloring BGR matrix notation

            cv2.putText(display_img, "LIVE 3D PHANTOM OVERLAY ACTIVE (640x480)", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("RealSense Live 3D Overlay View", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[*] Overlay pipeline terminated cleanly.")

if __name__ == "__main__":
    main() 