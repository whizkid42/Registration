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

def pick_point_on_cad(pcd):
    """
    Opens an interactive window to select a coordinate on the reference CAD model.
    Instructions: Hold SHIFT and LEFT-CLICK to select a point. Close window to confirm.
    """
    print("\n==================================================================")
    print(" INSTRUCTIONS FOR POINT SELECTION:")
    print(" 1. Hold [SHIFT] and [LEFT-CLICK] on the CAD surface to select a point.")
    print(" 2. A tiny selection sphere will appear on the chosen coordinate.")
    print(" 3. Close the window profile window (click X) to lock in your choice.")
    print("==================================================================\n")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Select Landmark Coordinate (Shift+LeftClick)")
    vis.add_geometry(pcd)
    vis.run()  
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    if not picked_indices:
        raise ValueError("No point was selected. System aborting tracking sequence.")
        
    selected_point_3d = np.asarray(pcd.points)[picked_indices[0]]
    print(f"[+] Stored Model Coordinate: X={selected_point_3d[0]:.4f}, Y={selected_point_3d[1]:.4f}, Z={selected_point_3d[2]:.4f}")
    return selected_point_3d

def main():
    if not os.path.exists(MODEL_PCD_PATH):
        raise FileNotFoundError(f"Missing dense reference point cloud at: {MODEL_PCD_PATH}")

    print("[*] Pre-loading reference model point cloud asset...")
    ref_model = o3d.io.read_point_cloud(MODEL_PCD_PATH)

    # -----------------------------------------------------------------
    # PHASE 1: CHOOSE TRACKING LANDMARK ON CAD MODEL
    # -----------------------------------------------------------------
    selected_point_model = pick_point_on_cad(ref_model)
    P_model = np.append(selected_point_model, 1.0) # Convert to homogeneous coordinate [X, Y, Z, 1.0]

    # -----------------------------------------------------------------
    # PHASE 2: INITIALIZE REALSENSE OPTICS & ARUCO DETECTORS
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # PHASE 3: LIVE RE-STREAM TARGET POINT TRACKING OVERLAY
    # -----------------------------------------------------------------
    print("\n==================================================================")
    print(" PHASE 3 START: Live Track Target Active. Press [q] to terminate.")
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
                    cv2.aruco.drawDetectedMarkers(display_img, corners, ids, borderColor=(0, 255, 0))
                    
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    Cam_T_Marker = np.eye(4)
                    Cam_T_Marker[:3, :3] = R_cam_marker
                    Cam_T_Marker[:3, 3] = tvec.squeeze()
                    
                    # Transform selected point: Model coordinate -> Marker coordinate -> Camera tracking frame
                    P_cam = Cam_T_Marker @ FINAL_AVERAGED_MATRIX @ P_model
                    X_c, Y_c, Z_c = P_cam[:3]

                    if Z_c > 0:  # Validate tracking point resides in front of optics glass structure
                        # Projection calculations: Mapping 3D coordinate space parameters into 2D display frames
                        pixel_homog = camera_matrix @ np.array([X_c, Y_c, Z_c])
                        u = int(pixel_homog[0] / pixel_homog[2])
                        v = int(pixel_homog[1] / pixel_homog[2])

                        # Draw highlighting overlay indicators onto the image arrays
                        if 0 <= u < display_img.shape[1] and 0 <= v < display_img.shape[0]:
                            cv2.circle(display_img, (u, v), 8, (0, 0, 255), -1)       # Target Node Center (Red)
                            cv2.circle(display_img, (u, v), 18, (0, 255, 255), 2)    # Outer Boundary Reticle Ring (Yellow)
                            cv2.putText(display_img, "TARGET LANDMARK", (u + 25, v + 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(display_img, "LIVE TRACKING ACTIVE (SAVED MASTER MATRIX)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("RealSense Live Tracking View", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[*] Tracking pipeline terminated cleanly.")

if __name__ == "__main__":
    main()