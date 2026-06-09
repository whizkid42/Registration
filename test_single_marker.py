import cv2
import numpy as np
import pyrealsense2 as rs
import os

def clear_console():
    """Wipes the terminal screen for smooth real-time metric reading."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_homogeneous_matrix(matrix, marker_id):
    """Outputs the formatted 4x4 matrix and its real-world metric conversion."""
    print("================================================================")
    print(f" TRACKING ACTIVE - Target Marker ID: {marker_id}")
    print("================================================================")
    print(" 4x4 HOMOGENEOUS TRANSFORMATION MATRIX [ Cam_T_Marker ]:")
    print("----------------------------------------------------------------")
    for row in matrix:
        print(f"  [ {row[0]:.4f},  {row[1]:.4f},  {row[2]:.4f}  |  {row[3]:.4f} ]")
    print("----------------------------------------------------------------")
    
    # Extract structural translations into centimeters
    x_cm = matrix[0, 3] * 100
    y_cm = matrix[1, 3] * 100
    z_cm = matrix[2, 3] * 100
    print(f" Position -> X: {x_cm:+.1f} cm | Y: {y_cm:+.1f} cm | Z (Distance): {z_cm:.1f} cm")
    print("================================================================\n")

def main():
    # 1. Configure Intel RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Match your existing stream settings
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    profile = pipeline.start(config)
    
    # 2. Extract Factory-Calibrated Intrinsics Matrix (K) directly from the device
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    camera_matrix = np.array([
        [intrinsics.fx, 0,             intrinsics.ppx],
        [0,             intrinsics.fy, intrinsics.ppy],
        [0,             0,             1             ]
    ], dtype=np.float32)
    
    # RealSense streams are internally rectified; lens distortion is set to zero
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # 3. Define 5x5 ArUco Parameters
    # Using DICT_5X5_50 (change to DICT_5X5_250 or DICT_5X5_1000 if your printed tag is different)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Optimize threshold parsing window for smaller physical dimensions (1.5cm)
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 4. Set Up 3D Object Space Coordinate Maps
    # Physical size = 1.5 cm = 0.015 meters
    marker_size = 0.015 
    half_s = marker_size / 2.0
    
    # Local coordinate model centered at the marker's midpoint origin [0,0,0]
    marker_obj_points = np.array([
        [-half_s,  half_s, 0],  # Top-Left corner
        [ half_s,  half_s, 0],  # Top-Right corner
        [ half_s, -half_s, 0],  # Bottom-Right corner
        [-half_s, -half_s, 0]   # Bottom-Left corner
    ], dtype=np.float32)

    print("\nRealSense Pipeline Streaming Active.")
    print("Bring your 1.5cm 5x5 ArUco marker into view...")
    print("Press 'q' in the camera window to terminate process.\n")

    try:
        while True:
            # Gather synchronized frame buffers
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame data to an OpenCV-compatible numpy array
            color_img = np.asanyarray(color_frame.get_data())
            output_img = color_img.copy()

            # Execute 2D marker corner extraction
            corners, ids, _ = detector.detectMarkers(color_img)

            if ids is not None and len(ids) > 0:
                # Isolate the primary marker parameters
                target_id = ids[0][0]
                marker_corners_2d = corners[0][0]

                # Run Iterative Perspective-n-Point solver to fetch 3D Pose
                success, rvec, tvec = cv2.solvePnP(
                    marker_obj_points, marker_corners_2d, 
                    camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Convert 3-axis rotation vector to a 3x3 rotation matrix via Rodrigues' formula
                    R, _ = cv2.Rodrigues(rvec)
                    
                    # Assemble the final 4x4 Homogeneous Matrix [Cam_T_Marker]
                    Cam_T_Marker = np.eye(4)
                    Cam_T_Marker[:3, :3] = R
                    Cam_T_Marker[:3, 3] = tvec.squeeze()

                    # Dynamic console logging refresh
                    clear_console()
                    print_homogeneous_matrix(Cam_T_Marker, target_id)

                    # Draw standard green border boundaries around the tag
                    cv2.aruco.drawDetectedMarkers(output_img, corners, ids, borderColor=(0, 255, 0))
                    
                    # Render a 3D coordinate frame axis directly on the marker center (length = 1.5 cm)
                    cv2.drawFrameAxes(output_img, camera_matrix, dist_coeffs, rvec, tvec, length=0.015, thickness=2)
            else:
                clear_console()
                print("Searching for 5x5 ArUco Marker (1.5cm dimension)...")

            # Update display view
            cv2.imshow("RealSense D435 - 5x5 Pose Solver", output_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Gracefully unmount resources
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()