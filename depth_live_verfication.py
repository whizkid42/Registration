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
VOXEL_DOWN_SIZE = 0.003 # 3mm point density for real-time projection mapping
# --------------------------------------

# --- YOUR INJECTED MASTER CALIBRATION MATRIX ---
FINAL_AVERAGED_MATRIX = np.array([
    [0.9982, -0.0359, -0.0477,  0.1487],
    [0.0318,  0.9959, -0.0845, -0.0031],
    [0.0506,  0.0828,  0.9953,  0.0360],
    [0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)
# -----------------------------------------------

def get_realsense_depth_intrinsics(profile):
    """Extracts factory calibration configurations directly from the DEPTH stream hardware."""
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    
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

    # Configure RealSense Hardware Pipeline Environment (640x480 @ 15 FPS)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    profile = pipeline.start(config)
    
    # --- CRITICAL UPDATE: Align color space stream to the DEPTH perspective ---
    align = rs.align(rs.stream.depth)
    camera_matrix, dist_coeffs, _ = get_realsense_depth_intrinsics(profile)
    
    # RealSense utility tool to color-map raw 16-bit depth vectors to 8-bit RGB spectrum frames
    colorizer = rs.colorizer()
    # --------------------------------------------------------------------------

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    half_s = MARKER_SIZE / 2.0
    marker_obj_pts = np.array([
        [-half_s, half_s, 0], [half_s, half_s, 0], 
        [half_s, -half_s, 0], [-half_s, -half_s, 0]
    ], dtype=np.float32)

    print("\n==================================================================")
    print(" PIPELINE ACTIVE: 3D CAD to DEPTH STREAM overlay active.")
    print(" Displaying colorized depth maps (640x480 @ 15 FPS).")
    print(" Press [q] inside the frame window to terminate.")
    print("==================================================================\n")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # Aligns frames so color images map perfectly into depth frame coordinates
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert frames to manageable matrix arrays
            color_img = np.asanyarray(color_frame.get_data())
            
            # --- COLORIZE THE RAW DEPTH DATA FOR DISPLAY ---
            colorized_depth_frame = colorizer.colorize(depth_frame)
            depth_img_color = np.asanyarray(colorized_depth_frame.get_data())
            display_img = depth_img_color.copy()
            # -----------------------------------------------
            
            # We must detect the ArUco marker using the aligned color frame image
            corners, ids, _ = detector.detectMarkers(color_img)
            
            if ids is not None and TARGET_ID in ids.flatten():
                idx = np.where(ids.flatten() == TARGET_ID)[0][0]
                
                # Using depth intrinsics matrix to solve PnP since color frame is warped to depth space
                success, rvec, tvec = cv2.solvePnP(marker_obj_pts, corners[idx][0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                
                if success:
                    # Draw ArUco boundary lines directly onto the depth background representation
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids, borderColor=(0, 255, 0))
                    
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    Cam_T_Marker = np.eye(4)
                    Cam_T_Marker[:3, :3] = R_cam_marker
                    Cam_T_Marker[:3, 3] = tvec.squeeze()
                    
                    # 1. Compute the full dynamic transformation path for this frame
                    Cam_T_Model = Cam_T_Marker @ FINAL_AVERAGED_MATRIX
                    
                    # 2. Multiply all CAD model coordinates simultaneously to jump to Depth Camera Frame
                    points_in_cam = Cam_T_Model @ cad_points_homog
                    
                    X_c = points_in_cam[0, :]
                    Y_c = points_in_cam[1, :]
                    Z_c = points_in_cam[2, :]
                    
                    # 3. Only keep points in front of the lens asset
                    front_indices = np.where(Z_c > 0.01)[0]
                    
                    if len(front_indices) > 0:
                        pts_to_project = np.vstack((X_c[front_indices], Y_c[front_indices], Z_c[front_indices]))
                        
                        # 4. Project coordinates onto depth space pixel sensor mesh dimensions
                        pixel_homog = camera_matrix @ pts_to_project
                        u_arr = (pixel_homog[0, :] / pixel_homog[2, :]).astype(np.int32)
                        v_arr = (pixel_homog[1, :] / pixel_homog[2, :]).astype(np.int32)
                        
                        # 5. Bounds check for 640x480 resolution constraints
                        valid_mask = (u_arr >= 0) & (u_arr < display_img.shape[1]) & \
                                     (v_arr >= 0) & (v_arr < display_img.shape[0])
                        
                        u_final = u_arr[valid_mask]
                        v_final = v_arr[valid_mask]
                        
                        # 6. Paint the projected CAD footprint points onto the colorized depth view (Solid White)
                        for u, v in zip(u_final, v_final):
                            display_img[v, u] = [255, 255, 255]  # White color overlay notation

            cv2.putText(display_img, "CAD OVERLAY ON DEPTH STREAM (640x480)", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("RealSense Live Depth Overlay View", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[*] Depth overlay pipeline terminated cleanly.")

if __name__ == "__main__":
    main()