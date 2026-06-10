
## **Spatial Registration**

This generalized pipeline provides real-time, cross-platform alignment of 3D CAD models onto physical targets using an Intel RealSense RGB-D camera. By utilizing a 5x5 ArUco marker as a dynamic coordinate anchor, the system computes the instantaneous 6-DOF pose of the target. It then transforms and wraps the digital point cloud precisely over the physical geometry across live 2D and 3D visualizers.

### **Mathematical Formulation and Coordinate Spaces**

The system operates across three distinct three-dimensional coordinate frames:
* **Camera Frame** (Cam): The moving origin centered at the physical optical lens of the Intel RealSense camera.
* **Marker Frame** (M): The localized origin situated at the center of the physical 2D ArUco tracking tag.
* **Model Frame** (O): The static native origin of your reference 3D CAD model.

#### **Transfomation Chain**
$\qquad T_{O}^{Cam}$ = $T_M^{Cam}$ x $T_O^M $

 Where,

  $T_M^{Cam}$ = Live Tracking Matrix - A dynamic $4 \times 4$ homogeneous transformation matrix calculated at every frame. The pipeline utilizes OpenCV’s solvePnP algorithm, taking known 2D image pixel points and mapping them against the marker's real-world dimensions to extract rotation ($\mathbf{R}$) and translation ($\mathbf{t}$) vectors.

  $T_{O}^{M}$ = Static Calibration Matrix - A rigid, fixed $4 \times 4$ matrix representing the transformation between the ArUco tag and the physical object. This is calculated once during an offline registration setup.

  **Note** - The Physical Object Frame and the CAD Model Frame are conceptually identical ($O$).Because the physical phantom is 3D printed directly from the digital CAD specifications, it acts as a 1:1 Digital Twin. It inherits the exact same fixed mathematical origin $(0,0,0)$ established in the design software. Treating the Model Frame and the Physical Object Frame as a single shared coordinate system ($O$) allows the matrix chain to transform the digital CAD coordinates and have them map onto the real-world physical object.

  ### **Implementation**

The development and execution environment for this pipeline is built on Windows OS and relies on hardware-accelerated spatial computing libraries. 

**Core Libraries** - pyrealsense2 , open3d , opencv-python , numpy

**Description of Scripts**

- **obj_to_pcd.py** (Asset Pre-processing): Converts standard .obj 3D mesh files of your CAD model into structured Point Cloud Data (.pcd) files, offering user-controlled configuration over vertex point density.

   - *Note* - CAD file units need to be in metres.

- **test_single_marker.py** (Fiducial Diagnostic Tool): An upfront validation script that isolates the RGB camera feed to confirm successful 5x5 ArUco marker localization and outputs the raw live camera-to-marker transform matrix ($^C\mathbf{T}_M$).

- **register_phantom.py** (The Mathematical Core): The primary registration engine executing the Iterative Closest Point (ICP) algorithm. It processes the source CAD file against real-world geometry to calculate a FINAL_AVERAGED_MATRIX averaged over multiple continuous user generated samples.

    - *Note* - THe physical object when placed in the center of the camera frame, gives better results.

- **verification.py** (Spatial Anchor Probe): Spins up an interactive window enabling the user to pin a specific coordinate on the reference CAD model. The script then dynamically tracks the object, mapping and projecting that isolated 3D point onto the live 2D camera stream via real-time pose calculations.

- **live_verification.py** (Pure 3D Overlay): Streams a continuous, low-latency 3D spatial canvas displaying the transformed CAD model positioned accurately over the tracking anchor.

- **visualiser_live_verification.py** (The Dual-Viewport Dashboard): The complete operational suite. It initializes a simultaneous, multi-threaded display rendering an interactive Open3D virtual environment side-by-side with the live 2D RealSense diagnostic video feed.

- **depth_live_verification.py** (Sensor-Fusion Validation): Projects and overlays the source CAD model's digital point cloud directly over the raw infrared depth stream data to verify geometrical surface alignment accuracy.
