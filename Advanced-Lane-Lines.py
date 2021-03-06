import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip

parameters = pickle.load(open('examples/camera_calibration_parameters', 'rb'))
mtx, dist = map(parameters.get, ('mtx', 'dist'))

# update at bird_eye function
ym_per_pix = 0
xm_per_pix = 0

# get by test images
lane_width_range = [2.8, 3.6]
before_left_fitx = np.array([])
before_right_fitx = np.array([])

def show_output_images(image_dict, cols=4, rows=5, figsize=(15, 13)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(rows * cols)

    for ax, index in zip(axes.flat, image_dict.keys()):
        ax.imshow(image_dict[index])
        ax.set_title(index)
        ax.axis('off')

def undistort(img):
    return  cv2.undistort(img, mtx, dist, None, mtx)

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    if orient == 'x':
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    elif orient == 'y':
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    binary_output = np.copy(sxbinary)
    return binary_output 

def space_thresh(img, space, thresh_min, thresh_max):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if space == 'H':
        img = hls_img[:, :, 0]
    elif space == 'L':
        img = hls_img[:, :, 1]
    elif space == 'S':
        img = hls_img[:, :, 2]

    binary = np.zeros_like(img)
    binary[(img>=thresh_min) & (img<=thresh_max)] = 1
    return binary

def lab_thresh(img, thresh_min, thresh_max):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_b = lab_img[:, :, 2]
    binary = np.zeros_like(lab_b)
    binary[(lab_b >= thresh_min) & (lab_b <= thresh_max)] = 1
    return binary

def rgb_thresh(img, thresh_min, thresh_max):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_img = rgb_img[:, :, 0]
    binary = np.zeros_like(r_img)
    binary[(r_img >= thresh_min) & (r_img <= thresh_max)] = 1
    return binary
   

def combine(img1, img2):
    binary = np.zeros_like(img1)
    binary[(img1 == 1) | (img2 == 1)] = 1
    return binary

def region_lines(img, vertices):
    copy_img = img.copy()
    for i in range(vertices.shape[1]-1):
        cv2.line(img, tuple(tuple(vertices[:, i])[0]), tuple(tuple(vertices[:, i+1])[0]), (0, 255, 0), 10)
    cv2.line(img, tuple(tuple(vertices[:, vertices.shape[1]-1])[0]), tuple(tuple(vertices[:, 0])[0]), (0, 255, 0), 10)
    return img

def bird_eye(img, src, offset):
    global ym_per_pix, xm_per_pix
    h, w = img.shape[:2]
    X, Y = img.shape[1], img.shape[0]
    dst = np.float32([
        (offset, Y),
        (offset, 0),
        (X-offset, 0),
        (X-offset, Y)
        ])
    ym_per_pix = 20 / Y
    xm_per_pix = 3.7 / (X-2*offset)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9
    margin = 100
    minpix = 50
    
    window_height = np.int(binary_warped.shape[0]//nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit, right_fit = (None, None)
    left_fit_m, right_fit_m = (None, None)
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    if len(rightx) != 0: 
        right_fit = np.polyfit(righty, rightx, 2)
        right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    
    return out_img, left_fit, right_fit, left_fit_m, right_fit_m


def calc_curve(left_fit_cr, right_fit_cr, y_eval=720):
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def draw_lane(img, left_fit, right_fit, Minv):
    global fail_flag, before_left_fitx, before_right_fitx
    y_max = img.shape[0]
    ploty = np.linspace(0, y_max -1, y_max)
    color_warp = np.zeros_like(img).astype(np.uint8)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    if fail_flag and before_left_fitx.size != 0:
        left_fitx = before_left_fitx
        right_fitx = before_right_fitx
    before_left_fitx = left_fitx
    before_right_fitx = right_fitx
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def draw_information(img, left_fit_m, right_fit_m):
    global lane_width_range, fail_flag
    y_max = img.shape[0] * ym_per_pix
    left, right = calc_curve(left_fit_m, right_fit_m) 

    vehicle_center = img.shape[1] * xm_per_pix / 2
    line_left = left_fit_m[0] * y_max ** 2 + left_fit_m[1] * y_max + left_fit_m[2]
    line_right = right_fit_m[0] * y_max ** 2 + right_fit_m[1] * y_max + right_fit_m[2]
    middle = (line_right + line_left)/2
    dist_from_center = middle - vehicle_center
    tmp_lane_width = line_right - line_left
    if tmp_lane_width < lane_width_range[0] or tmp_lane_width > lane_width_range[1]:
        fail_flag = 1
    else:
        fail_flag = 0
    if dist_from_center > 0:
        message = '{:.2f} m left of center'.format(dist_from_center)
    else:
        message = '{:.2f} m right of center'.format(-1*dist_from_center)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    cv2.putText(img, 'Left curvature: {} m'.format(int(left)), (50, 50), font, 2, fontColor, 2)
    cv2.putText(img, 'Right curvature: {} m'.format(int(right)), (50, 120), font, 2, fontColor, 2)
    cv2.putText(img, 'Vehicle is {} '.format(message), (50, 190), font, 2, fontColor, 2)
    cv2.putText(img, 'Lane width is {} '.format(tmp_lane_width), (50, 250), font, 2, fontColor, 2)
    
    return img

def pipeline(original_img, debugFlag=0):
    img = original_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imshape = img.shape
    undistort_img = undistort(img)
    if debugFlag:
        cv2.imwrite("./output_images/process/00_undistort_img.jpg", undistort_img)
    vertices = np.array([[(200, imshape[0]), (imshape[1]/2-80, imshape[0]*.65), 
        (imshape[1]/2+80, imshape[0]*.65), (1200, imshape[0])]], dtype=np.float32)
    warped_img, M, Minv = bird_eye(undistort_img, vertices, 200)
    if debugFlag:
        cv2.imwrite("./output_images/process/01_warped_img.jpg", warped_img)


    #lab_img = lab_thresh(warped_img, 0, 125)
    #if debugFlag:
    #    plt.imshow(lab_img, cmap='gray')
    #    plt.savefig("./output_images/process/lab_img.jpg")

    sobelx_img = abs_sobel_thresh(warped_img, 'x', 20, 70)
    if debugFlag:
        plt.imshow(sobelx_img, cmap='gray')
        plt.savefig("./output_images/process/sobelx_img.jpg")

    #s_binary = space_thresh(warped_img, 'L', 220, 255)
    #if debugFlag:
    #    plt.imshow(s_binary, cmap='gray')
    #    plt.savefig("./output_images/process/l_binary.jpg")

    r_img = rgb_thresh(warped_img, 200, 255)
    if debugFlag:
        plt.imshow(r_img, cmap='gray')
        plt.savefig("./output_images/process/rgb_thresh.jpg")

    combined_img = combine(sobelx_img, r_img)
    if debugFlag:
        plt.imshow(combined_img, cmap='gray')
        plt.savefig("./output_images/process/02_combined_img.jpg")

    out_img, left_fit, right_fit, left_fit_m, right_fit_m = find_lane_pixels(combined_img)
    if debugFlag:
        cv2.imwrite("./output_images/process/03_out_img.jpg", out_img)
    if left_fit is None or right_fit is None:
        return original_img
    else:
        out_img_with_info = draw_information(original_img, left_fit_m, right_fit_m)
        if debugFlag:
            cv2.imwrite("./output_images/process/04_out_img_with_info.jpg", out_img_with_info)
        out_img_with_lane = draw_lane(out_img_with_info, left_fit, right_fit, Minv)
        if debugFlag:
            cv2.imwrite("./output_images/process/05_out_img_with_lane.jpg", out_img_with_lane)
    return out_img_with_lane

def run_image():
    test_filenames = glob.glob('./test_images/*.jpg')
    test_img_dict = {}
    output_img_dict = {}
    for test_filename in test_filenames:
        test_img_dict[test_filename] = cv2.cvtColor(cv2.imread(test_filename), cv2.COLOR_BGR2RGB)
        output_img_dict[test_filename] = pipeline(test_img_dict[test_filename], 1)
    
    show_output_images(output_img_dict, cols=2, rows=3)

def run_video(input_name):
    video = VideoFileClip(input_name)
    clip = video.fl_image(pipeline) 
    clip.write_videofile("output_videos/"+input_name, audio=False)

img = cv2.imread('./test_images/challenges/challenge2.jpg')
#img = cv2.imread('./test_images/test3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
debugFlag = 1
output = pipeline(img, debugFlag)
#plt.imshow(output, cmap='gray')
#plt.savefig("./output_images/output_image.jpg")
#run_image()
#plt.savefig("./output_images/output_all_image.jpg")
#plt.show()
#video_name = 'challenge_video.mp4'
#video_name = 'project_video.mp4'
#run_video(video_name)
