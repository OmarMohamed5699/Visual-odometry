from pathlib import Path
import torch
import torchvision
import pykitti
torch.set_grad_enabled(False);
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pykitti
import spatialmath as sm
from spatialmath import SE3, SO3
from scipy.optimize import least_squares
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import pykitti.utils as utils

images = Path('assets')









device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


pose_stats = []






# Read the dataset sequence we just downloaded
basedir = 'Data'
date = '2011_09_30'
drive = '0018'
data = pykitti.raw(basedir, date, drive)





calib = data.calib
K = calib.K_cam2  
baseline = calib.b_rgb 
focal_length = K[0, 0]







# ================================== for feature extraction i will use superpoints method and for matcehing will use lightglue===================================





extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  
matcher = LightGlue(features='superpoint').eval().to(device)





def detect_features_and_match(img1, img2):

    # code from lightglue example notebook\
        
    image0 = torchvision.transforms.functional.pil_to_tensor(img1).type(torch.FloatTensor)/255
    image1 = torchvision.transforms.functional.pil_to_tensor(img2).type(torch.FloatTensor)/255

    # extract features and match them
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  

    kp1, kp2, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kp1, m_kp2 = kp1[matches[..., 0]], kp2[matches[..., 1]] 
    
    

    return m_kp1, m_kp2, matches






def _3D_coordenates_with_indices(m_kp1, m_kp2):
    
    fx = K[0,0]
    fy = K[1,1]  
    cx = K[0,2]
    cy = K[1,2]

    kp1_np = m_kp1.cpu().numpy() # convert datatype from tensor to numpy cause superpoint method return tensor
    kp2_np = m_kp2.cpu().numpy()  # convert datatype from tensor to numpy cause superpoint method return tensor
    disparity = kp1_np[:, 0] - kp2_np[:, 0]

    # filter out invalid disparities
    
    valid_range= (disparity > 0.5) & (disparity < 120) # FROM THE TESTS ON DATA BEFORE FOUND THE RANGE OF DISPARITY FROM 1.2 TO 116
    valid_disparity = disparity[valid_range]
    valid_kp1 = kp1_np[valid_range]
    valid_indices = np.where(valid_range)[0]  

    # Calculate 3D coordinates
    Z = (baseline * fx) / valid_disparity
    X = (valid_kp1[:, 0] - cx) * Z / fx  
    Y = (valid_kp1[:, 1] - cy) * Z / fy
    
    coordinates_3D = np.column_stack([X, Y, Z])
    
    return coordinates_3D, valid_kp1, valid_indices





    
    

def find_intersection(F2F_kpts, stereo_kpts, threshold=30.0): # F2F is fram to fram keypoint and we have stereo keypoint from left to right image


    if len(F2F_kpts) == 0 or len(stereo_kpts) == 0:
        return []
    
    # jsut to use gpu if and not cpu
    F2F = F2F_kpts.to(device)
    stereo = stereo_kpts.to(device)
    
    
    distances = torch.norm(
        
        F2F.unsqueeze(1) - stereo.unsqueeze(0), 
        dim=2
    )
    
    # Find best match for each temporal point
    min_distances, min_indices = torch.min(distances, dim=1)
    valid_mask = min_distances <= threshold
    
    # Extract matches
    matches = []
    valid_temporal_indices = torch.where(valid_mask)[0]
    
    for i in valid_temporal_indices:
        j = min_indices[i]
        dist = min_distances[i]
        matches.append((i.item(), j.item(), dist.item())) # we used .item  to convert from tensor to int
    
    return matches




# =============================================using 3D to 2D correspondences to find the camera pose with PnP RANSAC ==========================================

def solve_pnp(object_points_3d, image_points_2d, K):
    
    
    object_points = object_points_3d.astype(np.float32)
    image_points = image_points_2d.astype(np.float32)
    camera_matrix = K.astype(np.float32)
    
    # we assume no lens distortion for now
    dist_coeffs = np.zeros((4,1))
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        
        
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        reprojectionError=2.0, # 2.0 because SuperPoint features have ~1-2 pixel localization accuracy
        iterationsCount=1000,
        confidence=0.99
    )
    
    if not success:
        
        print("PnP failed!")
        return None, None, None, None  
    
    R, _ = cv2.Rodrigues(rvec) # converting rotation vector to rotation matrix

    
    return R, tvec, inliers, rvec  





def pose_to_vector(R, t):
    
    rvec, _ = cv2.Rodrigues(R)
    
    return np.concatenate([t.flatten(), rvec.flatten()])





def vector_to_pose(vec):
   
   
    t = vec[:3].reshape(3, 1)
    rvec = vec[3:6].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    
    
    return R, t





def project_points(landmarks_3d, pose_vec, K):
    
    
    R, t = vector_to_pose(pose_vec)
    points_cam = (R @ landmarks_3d.T + t).T
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    points_2d[:, 0] = points_2d[:, 0] * K[0, 0] + K[0, 2]
    points_2d[:, 1] = points_2d[:, 1] * K[1, 1] + K[1, 2]
    return points_2d






def bundle_adjustment_residuals(params, observations, K, n_poses, n_landmarks):
    
    
    pose_params = params[:n_poses * 6].reshape(n_poses, 6)
    landmark_params = params[n_poses * 6:].reshape(n_landmarks, 3)
    
    residuals = []
    for obs in observations:
        frame_id = obs['frame_id']
        landmark_ids = obs['landmark_ids']
        observed_points = obs['image_points']
        
        if frame_id >= n_poses:
            continue
            
        valid_landmark_ids = [lid for lid in landmark_ids if lid < n_landmarks]
        if len(valid_landmark_ids) == 0:
            continue
            
        projected = project_points(
            landmark_params[valid_landmark_ids], 
            pose_params[frame_id], 
            K
        )
        
        errors = projected - observed_points[:len(valid_landmark_ids)]
        residuals.extend(errors.flatten())
    
    return np.array(residuals)








def local_bundle_adjustment(poses, landmarks, observations, K, window_size=10): 


    if len(poses) < 3 or len(landmarks) == 0:
        return poses, landmarks


    # we select recent poses
    start_idx = max(0, len(poses) - window_size)
    local_poses = poses[start_idx:]


    # we select observations corresponding to local poses
    local_observations = [obs for obs in observations if obs['frame_id'] >= start_idx]

    if len(local_observations) < 2:
        return poses, landmarks


    num_residuals = sum(len(obs['image_points']) for obs in local_observations) * 2  # x and y
    num_variables = len(local_poses) * 6 + len(landmarks) * 3  # 6 per pose, 3 per landmark


    if num_residuals < num_variables:
        print(f"skip bundle adjustment because not enough residuals ({num_residuals} < {num_variables})")
        return poses, landmarks

    #  poses to vectors
    pose_vectors = [pose_to_vector(p['R'], p['t']) for p in local_poses]
    initial_params = np.concatenate([np.array(pose_vectors).flatten(), np.array(landmarks).flatten()])

    try:
        # TRF solver
        result = least_squares(
            bundle_adjustment_residuals,
            initial_params,
            args=(local_observations, K, len(local_poses), len(landmarks)),
            method='trf',  
            max_nfev=100
        )

        # Extract poses
        optimized_poses = result.x[:len(local_poses) * 6].reshape(len(local_poses), 6)
        optimized_landmarks = result.x[len(local_poses) * 6:].reshape(len(landmarks), 3)

        # update poses and landmarks
        for i, pose_vec in enumerate(optimized_poses):
            R, t = vector_to_pose(pose_vec)
            poses[start_idx + i]['R'] = R
            poses[start_idx + i]['t'] = t

        landmarks = optimized_landmarks.tolist()
        print(f"Bundle adjustment has optimized {len(local_poses)} poses.")

    except Exception as e:
        print(f"faild: {e}")

    return poses, landmarks







def visual_odometry_sequence(data, start_frame, num_frames, K, baseline):
    # Run visual odometry on a sequence of frames
    
    
    all_poses = []
    trajectory = []
    pose_stats = []  
    landmarks = []
    observations = []
    landmark_counter = 0
    
    # Initialize cumulative pose
    cumulative_R = np.eye(3)
    cumulative_t = np.zeros((3, 1))
    
    # Store initial position
    initial_position = np.array([0.0, 0.0, 0.0])
    trajectory.append(initial_position)
    
    # add intial pose for bundel adjustment
    all_poses.append({'R': np.eye(3), 't': np.zeros((3, 1))})
    
    print(f"Processing {num_frames-1} frame pairs starting from frame {start_frame}")
    
    for frame_idx in range(start_frame, start_frame + num_frames - 1):
        print(f"\n Processing frames {frame_idx} to {frame_idx + 1} ")
        
        try:
            # Get consecutive frames
            left_current, right_current = data.get_rgb(frame_idx) #
            left_next, _ = data.get_rgb(frame_idx + 1)
            
            # temporal matching , current to next
            temporal_current, temporal_next, _ = detect_features_and_match(left_current, left_next)
            
            # stereo matching current left to current right
            stereo_curr, stereo_right, _ = detect_features_and_match(left_current, right_current)
            
            # get the 3D points and valid stereo keypoints
            Three_D_points, valid_stereo_kpts, valid_stereo_indices = _3D_coordenates_with_indices(stereo_curr, stereo_right)
            
            # find intersection
            matches = find_intersection(temporal_current, torch.tensor(valid_stereo_kpts), threshold=30.0)
            
            if len(matches) < 50:  # minimum matches 
                print(f"Not enough matches: {len(matches)} < 50. will skip the frame.")
                continue
            
            # extract correspondences
            temporal_indices = [match[0] for match in matches]
            valid_stereo_match_indices = [match[1] for match in matches]
            
            object_points_3d = Three_D_points[valid_stereo_match_indices]
            image_points_2d = temporal_next[temporal_indices].cpu().numpy()
            
            # solve PnP
            R, tvec, inliers, rvec = solve_pnp(object_points_3d, image_points_2d, K)
            
            if R is None:
                print("PnP failed. Skipping frame.")
                continue
            
            # check for enough inliers
            inlier_ratio = len(inliers) / len(matches)
            if inlier_ratio < 0.3:  # 30% minimum inlier ratio
                print(f"Too few inliers: {inlier_ratio:.2%}. Skipping frame.")
                continue
            
            
            current_landmarks = object_points_3d[inliers.flatten()]

            # Limit landmarks due to memory limits
            max_landmarks_per_frame = 50  
            if len(current_landmarks) > max_landmarks_per_frame:
                
                
                # Keep landmarks with best matches
                distances = [match[2] for match in matches]
                inlier_distances = [distances[i] for i in range(len(distances)) if i in inliers.flatten()]
                sorted_indices = np.argsort(inlier_distances)[:max_landmarks_per_frame]
                current_landmarks = current_landmarks[sorted_indices]


            landmark_ids = list(range(landmark_counter, landmark_counter + len(current_landmarks)))
            landmarks.extend(current_landmarks.tolist())
            landmark_counter += len(current_landmarks)

            #  limit total landmarks in memory
            max_total_landmarks = 1000  # maximum landmarks 
            if len(landmarks) > max_total_landmarks:
                
                landmarks = landmarks[-max_total_landmarks:]   # keep only most recent landmarks

                landmark_ids = [i for i in landmark_ids if i >= landmark_counter - max_total_landmarks]  # update landmark_ids accordingly

            
            
            observations.append({
                'frame_id': len(all_poses),
                'landmark_ids': landmark_ids,
                'image_points': image_points_2d[inliers.flatten()]
            })
            
            # update cumulative pose
            cumulative_R = R @ cumulative_R 
            cumulative_t = R @ cumulative_t + tvec
            
            # Convert to world position
            world_position = -cumulative_R.T @ cumulative_t
            trajectory.append(world_position.flatten())
            
            # ADD THIS BACK - POSE STORAGE FOR BUNDLE ADJUSTMENT
            all_poses.append({'R': cumulative_R.copy(), 't': cumulative_t.copy()})
            
            # local bundle adjustment every 10 frames
            if len(all_poses) % 10 == 0 and len(all_poses) > 5:
                print("performing local bundle adjustment")
                all_poses, landmarks = local_bundle_adjustment(
                    all_poses, landmarks, observations, K, window_size=10
                 )
                
                # update trajectory from optimized poses
                for i in range(max(0, len(all_poses) - 10), len(all_poses)):
                    if i < len(trajectory):
                        world_pos = -all_poses[i]['R'].T @ all_poses[i]['t']
                        trajectory[i] = world_pos.flatten()
                
            # Store pose info in dectionary
            translation_magnitude = np.linalg.norm(tvec)
            rotation_angle = np.linalg.norm(rvec) * 180 / np.pi
            
            pose_info = {
                'frame': frame_idx + 1,
                'inliers': len(inliers),
                'total_matches': len(matches),
                'inlier_ratio': inlier_ratio,
                'translation_mag': translation_magnitude,
                'rotation_angle': rotation_angle,
                'position': world_position.flatten()
            }
            pose_stats.append(pose_info) 
            
        except Exception as e:
            
            print(f"Error processing frame {frame_idx}: {e}")
            continue
    
    
    final_trajectory = []
    for pose in all_poses:
        if 'R' in pose and 't' in pose:
            world_pos = -pose['R'].T @ pose['t']
            final_trajectory.append(world_pos.flatten())

    trajectory = np.array(final_trajectory)
    
    
    
    return trajectory, pose_stats, all_poses
        
        



trajectory, pose_info, all_poses = visual_odometry_sequence(data, start_frame=0, num_frames=2761, K=K, baseline=baseline)






# ========================================================================= Plotting =============================================================




trajectory = np.column_stack([trajectory[:, 0], trajectory[:, 2]])
gt_x = [oxts.T_w_imu[0,3] for oxts in data.oxts[:len(trajectory)]]
gt_y = [oxts.T_w_imu[1,3] for oxts in data.oxts[:len(trajectory)]]

plt.figure(figsize=(10, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='VO')
plt.plot(gt_x, gt_y, 'r-', linewidth=2, label='Ground Truth')
plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Visual Odometry vs Ground Truth')
plt.legend()
plt.grid(True)
plt.show()

min_len = min(len(trajectory), len(gt_x))
gt_xy = np.column_stack([gt_x[:min_len], gt_y[:min_len]])
error = np.linalg.norm(trajectory[:min_len] - gt_xy, axis=1)
traj_xy = trajectory[:min_len]

print(f'Mean ATE: {np.mean(error):.2f} m')
print(f'Median ATE: {np.median(error):.2f} m')
print(f'Max ATE: {np.max(error):.2f} m')
print(f'RMSE: {np.sqrt(np.mean((traj_xy - gt_xy)**2)):.2f} m')


plt.figure(figsize=(10, 8))
plt.plot(error, 'b-', linewidth=2, label='error')
plt.xlabel('frames')
plt.ylabel('error (m)')
plt.title('Error over FRAMES')
plt.grid(True)
plt.show()



















