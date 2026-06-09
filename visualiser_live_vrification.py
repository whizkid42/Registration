import os
import copy
import cv2
import time
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# --- USER CONFIGURATION PARAMETERS ---
MODEL_PCD_PATH = r"C:\Users\Admin\Documents\phantom_registration\dense_phantom_perfect.pcd"
MARKER_SIZE = 0.05     
TARGET_ID = 0           
# --------------------------------------

# --- YOUR INJECTED MASTER CALIBRATION MATRIX ---
FINAL_AVERAGED_MATRIX = np.array([
    [0.9982, -0.0359, -0.0477,  0.1487],
    [0.0318,  0.9959, -0.0845, -0.0031],
    [0.0506,  0.0828,  0.9953,  0.0360],
    [0.0000,  0.0000,  0.0000,  1.0000]
], dtype=np.float32)

def main():
    if not os.path.exists(MODEL_PCD_PATH):
        raise FileNotFoundError(f"Missing point cloud asset at: {MODEL_PCD_PATH}")

    print("[*] Loading reference model point cloud asset...")
    ref_model = o3d.io.read_point_cloud(MODEL_PCD_PATH)

    bbox = ref_model.get_axis_aligned_bounding_box()

    print("\n========== MODEL INFO ==========")
    print("Min Bound:", bbox.min_bound)
    print("Max Bound:", bbox.max_bound)
    print("Extent:", bbox.get_extent())
    print("Center:", ref_model.get_center())
    print("================================\n")

    ref_model.paint_uniform_color([0.0, 1.0, 0.0]) # Solid neon green for CAD

    # 1. HARDWARE RESET TO CLEAR USB CONTROLLER CACHE
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > 0:
        print(f"[*] Found device: {devices[0].get_info(rs.camera_info.name)}. Forcing hardware power-cycle...")
        devices[0].hardware_reset()
        time.sleep(3.5) # Safe rest for the device to wake back up on the USB bus
    else:
        print("[!] No physical RealSense device detected on the USB bus! Attempting to proceed anyway...")

    # 2. CONFIGURE PIPELINE ENGINE & DEFINITIONS
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    profile = pipeline.start(config)
    
    # --- HERE IS THE FIX: Properly defining the alignment variable ---
    align = rs.align(rs.stream.color)
    # ------------------------------------------------------------------
    
    # Extract Camera Intrinsic Parameters for ArUco tracking
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

    # Native RealSense Point Cloud hardware accelerator engine
    rs_pc = rs.pointcloud()

    # Setup Open3D Window Thread 
    print("[*] Launching Open3D Renderer Thread...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live RealSense 3D Environment (30 FPS)", width=800, height=600)

    live_scene_cloud = o3d.geometry.PointCloud()
    live_cad_cloud = o3d.geometry.PointCloud()

    vis.add_geometry(live_scene_cloud)
    vis.add_geometry(live_cad_cloud)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)

    # Setup ArUco Tracking
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    half_s = MARKER_SIZE / 2.0
    marker_obj_pts = np.array([[-half_s, half_s, 0], [half_s, half_s, 0], [half_s, -half_s, 0], [-half_s, -half_s, 0]], dtype=np.float32)

    print("\n==================================================================")
    print(" PIPELINE ACTIVE: Native RealSense Processing + Open3D Window")
    print(" - Left-Click & Drag : Rotate Viewport")
    print(" - Right-Click & Drag: Pan Scene")
    print(" Press [q] inside the 2D window or close the 3D frame window to exit.")
    print("==================================================================\n")

    try:
        while True:
            if not vis.poll_events():
                break

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            
            # Generate 3D pointcloud via RealSense hardware context
            rs_pc.map_to(color_frame)
            points = rs_pc.calculate(depth_frame)
            
            # Zero-copy pointer manipulation for performance layout
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Distance filter threshold (Trims background past 1.5m)
            valid_mask = (vtx[:, 2] > 0.01) & (vtx[:, 2] < 1.5)
            vtx_filtered = vtx[valid_mask]
            
            colors_bgr = cv2.resize(color_img, (640, 480))
            colors_rgb = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
            colors_filtered = colors_rgb[valid_mask]

            live_scene_cloud.points = o3d.utility.Vector3dVector(vtx_filtered)
            live_scene_cloud.colors = o3d.utility.Vector3dVector(colors_filtered)

            # Look for the tracker target
            corners, ids, _ = detector.detectMarkers(color_img)

            print("\n----------------------------")
            print("Detected IDs:", ids)
            
            if ids is not None and TARGET_ID in ids.flatten():

                idx = np.where(ids.flatten() == TARGET_ID)[0][0]

                success, rvec, tvec = cv2.solvePnP(
                    marker_obj_pts,
                    corners[idx][0],
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                print("solvePnP success:", success)

                if success:

                    distance = np.linalg.norm(tvec)

                    print("Marker distance:", distance)
                    print("tvec:", tvec.flatten())

                    cv2.aruco.drawDetectedMarkers(
                        color_img,
                        corners,
                        ids,
                        borderColor=(0, 255, 0)
                    )

                    R_cam_marker, _ = cv2.Rodrigues(rvec)

                    Cam_T_Marker = np.eye(4)
                    Cam_T_Marker[:3, :3] = R_cam_marker
                    Cam_T_Marker[:3, 3] = tvec.squeeze()

                    print("Cam_T_Marker:")
                    print(Cam_T_Marker)

                    Cam_T_Model = Cam_T_Marker @ FINAL_AVERAGED_MATRIX

                    print("Cam_T_Model translation:")
                    print(Cam_T_Model[:3, 3])

                    transformed_cad = copy.deepcopy(ref_model)
                    transformed_cad.transform(Cam_T_Model)

                    live_cad_cloud.points = transformed_cad.points
                    live_cad_cloud.colors = transformed_cad.colors

            else:

                print("TARGET MARKER NOT DETECTED")

                live_cad_cloud.points = o3d.utility.Vector3dVector()
                live_cad_cloud.colors = o3d.utility.Vector3dVector()

            # Tick render loop refresh rate updates
            vis.update_geometry(live_scene_cloud)
            vis.update_geometry(live_cad_cloud)
            vis.update_renderer()

            # Show standard 2D control status feed
            cv2.imshow("RealSense Target Frame Diagnostic Feed", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("[*] Stream pipelines closed cleanly.")

if __name__ == "__main__":
    main()