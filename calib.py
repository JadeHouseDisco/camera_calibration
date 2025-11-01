import cv2 as cv
import glob
import itertools
import numpy as np
import sys
from scipy import linalg
import yaml
import os

#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}
_camera_device_map = {}
_camera_index_map = {}
_camera_order = []

#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):

    global calibration_settings, _camera_device_map, _camera_index_map, _camera_order

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f) or {}

    cameras = calibration_settings.get('cameras')
    if not isinstance(cameras, list):
        print('"cameras" list was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()

    if len(cameras) < 2:
        print('At least two cameras must be defined in the settings file for stereo calibration')
        quit()

    _camera_device_map = {}
    _camera_index_map = {}
    _camera_order = []

    for idx, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            print('Each camera entry must be a mapping with "name" and "device_id" fields')
            quit()

        name = camera.get('name')
        device_id = camera.get('device_id')

        if name is None or device_id is None:
            print('Each camera entry must define both "name" and "device_id" fields')
            quit()

        name = str(name)
        try:
            device_id = int(device_id)
        except (TypeError, ValueError):
            print(f'Camera "{name}" has an invalid "device_id". It must be an integer value')
            quit()

        if name in _camera_device_map:
            print(f'Duplicate camera name found: "{name}". Camera names must be unique')
            quit()

        camera['name'] = name
        camera['device_id'] = device_id

        _camera_device_map[name] = device_id
        _camera_index_map[name] = idx
        _camera_order.append(name)


def get_camera_device_id(camera_name):
    try:
        return _camera_device_map[camera_name]
    except KeyError:
        raise KeyError(f'Camera "{camera_name}" is not defined in calibration_settings.yaml')


def get_camera_index(camera_name):
    try:
        return _camera_index_map[camera_name]
    except KeyError:
        raise KeyError(f'Camera "{camera_name}" is not defined in calibration_settings.yaml')


def get_configured_camera_names():
    return list(_camera_order)


def _ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def _create_video_capture(camera_name):
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap = cv.VideoCapture(get_camera_device_id(camera_name))
    cap.set(3, width)
    cap.set(4, height)
    return cap


#Open camera stream and save frames
def save_frames_single_camera(camera_name):

    #create frames directory
    _ensure_directory('frames')

    #get settings
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    #open video stream and change resolution.
    #Note: if unsupported resolution is used, this does NOT raise an error.
    cap = _create_video_capture(camera_name)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
    
        ret, frame = cap.read()
        if ret == False:
            #if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cap.release()
    cv.destroyAllWindows()


def save_frames_pair(reference_camera_name, target_camera_name):

    _ensure_directory('frames_pair')
    pair_dir = os.path.join('frames_pair', f'{reference_camera_name}_vs_{target_camera_name}')
    _ensure_directory(pair_dir)

    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = calibration_settings['stereo_calibration_frames']

    cap_reference = _create_video_capture(reference_camera_name)
    cap_target = _create_video_capture(target_camera_name)

    header_text = f'{reference_camera_name} vs {target_camera_name}'
    cooldown = cooldown_time
    start = False
    saved_count = 0

    try:
        while True:

            ret_reference, frame_reference = cap_reference.read()
            ret_target, frame_target = cap_target.read()

            if not ret_reference or not ret_target:
                print(f'Cameras {reference_camera_name} and {target_camera_name} not returning video data. Exiting...')
                quit()

            frame_reference_small = cv.resize(frame_reference, None, fx=1./view_resize, fy=1./view_resize)
            frame_target_small = cv.resize(frame_target, None, fx=1./view_resize, fy=1./view_resize)

            cv.putText(frame_reference_small, header_text, (50, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame_target_small, header_text, (50, 40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            if not start:
                instruction_text = 'Press SPACEBAR to start collecting frames'
                cv.putText(frame_reference_small, instruction_text, (50, 80), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv.putText(frame_target_small, instruction_text, (50, 80), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            if start:
                cooldown -= 1
                cooldown_text = f'Cooldown ({header_text}): {cooldown}'
                count_text = f'Saved frames: {saved_count}/{number_to_save}'
                cv.putText(frame_reference_small, cooldown_text, (50, 90), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv.putText(frame_reference_small, count_text, (50, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv.putText(frame_target_small, cooldown_text, (50, 90), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                cv.putText(frame_target_small, count_text, (50, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                if cooldown <= 0:
                    reference_save = os.path.join(pair_dir, f'{reference_camera_name}_{saved_count}.png')
                    target_save = os.path.join(pair_dir, f'{target_camera_name}_{saved_count}.png')
                    cv.imwrite(reference_save, frame_reference)
                    cv.imwrite(target_save, frame_target)
                    saved_count += 1
                    cooldown = cooldown_time

            cv.imshow(f'{reference_camera_name}_view', frame_reference_small)
            cv.imshow(f'{target_camera_name}_view', frame_target_small)
            k = cv.waitKey(1)

            if k == 27:
                quit()

            if k == 32:
                start = True

            if saved_count == number_to_save:
                break
    finally:
        cap_reference.release()
        cap_target.release()
        cv.destroyAllWindows()


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    #NOTE: images_prefix contains camera name: "frames/<camera_name>*".
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


#open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):
    save_frames_pair(camera0_name, camera1_name)


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera_calibrations, _zshift = 50., overlay_pairs = None):

    if not isinstance(camera_calibrations, dict) or not camera_calibrations:
        raise ValueError('camera_calibrations must be a non-empty mapping of name to (cmtx, dist, R, T) tuples')

    overlay_pairs = overlay_pairs or []

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coordinate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    colors = [(0,0,255), (0,255,0), (255,0,0)]  # RGB colors to indicate XYZ axes respectively

    prepared_cameras = {}
    open_windows = set()

    def _normalize_translation_vector(tvec):
        tvec = np.asarray(tvec, dtype = np.float64)
        if tvec.size != 3:
            raise ValueError('Translation vectors must contain exactly three elements')
        return tvec.reshape(3, 1)

    def _prepare_overlay_frame(frame_a, frame_b):
        if frame_a.shape[:2] != frame_b.shape[:2]:
            target_height = min(frame_a.shape[0], frame_b.shape[0])
            target_width = min(frame_a.shape[1], frame_b.shape[1])
            frame_a = cv.resize(frame_a, (target_width, target_height))
            frame_b = cv.resize(frame_b, (target_width, target_height))
        return cv.addWeighted(frame_a, 0.5, frame_b, 0.5, 0)

    # Prepare per-camera data
    for camera_name, data in camera_calibrations.items():
        if not isinstance(data, (tuple, list)) or len(data) != 4:
            raise ValueError(f'Calibration data for camera "{camera_name}" must be a 4-tuple of (cmtx, dist, R, T)')

        cmtx = np.asarray(data[0], dtype = np.float64)
        dist = np.asarray(data[1], dtype = np.float64)
        R = np.asarray(data[2], dtype = np.float64)
        T = _normalize_translation_vector(data[3])

        P = get_projection_matrix(cmtx, R, T)
        pixel_points = []
        for _p in draw_axes_points:
            X = np.array([_p[0], _p[1], _p[2], 1.])
            uv = P @ X
            uv = np.array([uv[0], uv[1]])/uv[2]
            pixel_points.append(uv)
        pixel_points = np.array(pixel_points)

        cap = _create_video_capture(camera_name)
        if not cap.isOpened():
            raise RuntimeError(f'Could not open video stream for camera "{camera_name}"')

        prepared_cameras[camera_name] = {
            'capture': cap,
            'pixel_points': pixel_points,
        }

    # Filter overlay pairs to those that exist, otherwise fall back to all combinations
    camera_names = list(prepared_cameras.keys())
    valid_overlay_pairs = []
    for pair in overlay_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        a, b = pair
        if a in prepared_cameras and b in prepared_cameras and a != b:
            valid_overlay_pairs.append((a, b))
    if not valid_overlay_pairs and len(camera_names) > 1:
        valid_overlay_pairs = list(itertools.combinations(camera_names, 2))

    display_modes = ['all']
    if valid_overlay_pairs:
        display_modes.append('overlay')
    display_mode_index = 0
    active_overlay_index = 0

    try:
        while True:
            frames_with_axes = {}
            desired_windows = set()

            for camera_name, info in prepared_cameras.items():
                cap = info['capture']
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f'Video stream not returning frame data for camera "{camera_name}"')

                frame_with_axes = frame.copy()
                origin = tuple(info['pixel_points'][0].astype(np.int32))
                for col, _p in zip(colors, info['pixel_points'][1:]):
                    point = tuple(_p.astype(np.int32))
                    cv.line(frame_with_axes, origin, point, col, 2)

                frames_with_axes[camera_name] = frame_with_axes

            display_mode = display_modes[display_mode_index]

            if display_mode == 'overlay' and valid_overlay_pairs:
                cam_a, cam_b = valid_overlay_pairs[active_overlay_index]
                overlay_window = f'{cam_a}_vs_{cam_b}_overlay'
                overlay_frame = _prepare_overlay_frame(frames_with_axes[cam_a], frames_with_axes[cam_b])
                cv.imshow(overlay_window, overlay_frame)
                desired_windows.add(overlay_window)
            else:
                for camera_name, frame_with_axes in frames_with_axes.items():
                    window_name = f'{camera_name}_view'
                    cv.imshow(window_name, frame_with_axes)
                    desired_windows.add(window_name)

            for window in open_windows - desired_windows:
                cv.destroyWindow(window)
            open_windows = desired_windows

            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == ord('m') and len(display_modes) > 1:
                display_mode_index = (display_mode_index + 1) % len(display_modes)
            if k == ord('n') and display_modes[display_mode_index] == 'overlay' and valid_overlay_pairs:
                active_overlay_index = (active_overlay_index + 1) % len(valid_overlay_pairs)
            if k == ord('p') and display_modes[display_mode_index] == 'overlay' and valid_overlay_pairs:
                active_overlay_index = (active_overlay_index - 1) % len(valid_overlay_pairs)
    finally:
        for info in prepared_cameras.values():
            info['capture'].release()
        cv.destroyAllWindows()

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1


def _write_matrix(outf, label, matrix):
    outf.write(f'{label}:\n')
    arr = np.asarray(matrix)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for row in arr:
        outf.write(' '.join(str(value) for value in row) + '\n')


def save_extrinsic_calibration_parameters(camera_extrinsics, prefix = ''):

    if not isinstance(camera_extrinsics, dict):
        raise TypeError('camera_extrinsics must be a mapping of camera name to (R, T) tuples')

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    for camera_name, extrinsics in camera_extrinsics.items():
        if not isinstance(extrinsics, (list, tuple)) or len(extrinsics) != 2:
            raise ValueError(f'Extrinsics for camera "{camera_name}" must be a (R, T) tuple')

        R, T = extrinsics
        camera_rot_trans_filename = os.path.join('camera_parameters', f"{prefix}{camera_name}_rot_trans.dat")
        with open(camera_rot_trans_filename, 'w') as outf:
            _write_matrix(outf, 'R', R)
            _write_matrix(outf, 'T', T)

    return camera_extrinsics

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()

    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])

    camera_names = get_configured_camera_names()
    primary_camera_name, secondary_camera_name = camera_names[0], camera_names[1]


    """Step1. Save calibration frames for single cameras"""
    for camera_name in camera_names:
        save_frames_single_camera(camera_name)


    """Step2. Obtain camera intrinsic matrices and save them"""
    camera_intrinsics = {}
    for camera_name in camera_names:
        images_prefix = os.path.join('frames', f'{camera_name}_*')
        cmtx, dist = calibrate_camera_for_intrinsic_parameters(images_prefix)
        save_camera_intrinsics(cmtx, dist, camera_name)
        camera_intrinsics[camera_name] = (cmtx, dist)

    cmtx0, dist0 = camera_intrinsics[primary_camera_name]


    """Step3. Save calibration frames for both cameras simultaneously"""
    for target_camera_name in camera_names[1:]:
        save_frames_pair(primary_camera_name, target_camera_name)


    """Step4. Use paired calibration pattern frames to obtain rotation and translation between the primary camera and each target camera"""
    camera_extrinsics = {}

    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    camera_extrinsics[primary_camera_name] = (R0, T0)

    for target_camera_name in camera_names[1:]:
        cmtx_target, dist_target = camera_intrinsics[target_camera_name]
        pair_dir = os.path.join('frames_pair', f'{primary_camera_name}_vs_{target_camera_name}')
        frames_prefix_c0 = os.path.join(pair_dir, f'{primary_camera_name}_*')
        frames_prefix_c1 = os.path.join(pair_dir, f'{target_camera_name}_*')
        R, T = stereo_calibrate(cmtx0, dist0, cmtx_target, dist_target, frames_prefix_c0, frames_prefix_c1)
        camera_extrinsics[target_camera_name] = (R, T)


    """Step5. Save calibration data where the primary camera defines the world space origin."""
    save_extrinsic_calibration_parameters(camera_extrinsics)

    if len(camera_names) > 1:
        calibration_data = {}
        for camera_name in camera_names:
            cmtx, dist = camera_intrinsics[camera_name]
            R, T = camera_extrinsics[camera_name]
            calibration_data[camera_name] = (cmtx, dist, R, T)

        #check your calibration makes sense
        check_calibration(calibration_data, _zshift = 60.)


    """Optional. Define a different origin point and save the calibration data"""
    # #get the world to camera0 rotation and translation
    # R_W0, T_W0 = get_world_space_origin(cmtx0, dist0, os.path.join(primary_pair_dir, f'{primary_camera_name}_4.png'))
    # #get rotation and translation from world directly to camera1
    # R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,
    #                                           cmtx1, dist1, R1, T1,
    #                                           os.path.join(primary_pair_dir, f'{primary_camera_name}_4.png'),
    #                                           os.path.join(primary_pair_dir, f'{secondary_camera_name}_4.png'),)

    # #save rotation and translation parameters to disk
    # save_extrinsic_calibration_parameters(primary_camera_name, R_W0, T_W0, secondary_camera_name, R_W1, T_W1, prefix = 'world_to_')

