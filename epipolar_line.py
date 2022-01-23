import pickle
import glob
import matplotlib.pyplot as plt
from epipolar_geometry import compute_fundamental, plot_epipolar_line 
from vis import show3Dpose, show2Dpose
import numpy as np
from PIL import Image
from epipolar_geometry import *



json_file_path = "annot/h36m_validation.pkl"
with open(json_file_path, 'rb') as f:
    data = pickle.load(f)

# 전체이미지 개수 : 1559752

root = 'images'

subject_all_folder = glob.glob(root + '/*')

S_all_image_folder = []

for S in subject_all_folder:

    S_action = glob.glob(S + '/*')
    S_all_image_folder.append([glob.glob(A) for A in S_action])
    
   



"""
image: 이미지 파일명
joints_2d : 2D 조인트 좌표
joints_3d : 3D 조인트 좌표
joints_3d_camera: ??? joints_3d랑 같음
joints_vis: (17,3)이고 모두 1
video_id : 비디오 번호
image_id: 이미지 번호
subject: 배우번호
action : 액션 번호
subaction: 서브액션 번호
camera_id : 카메라 번호
source : h36m
camera: 카메라 파라미터
    R: 로테이션 매트릭스 (3,3)
    T: 트렌스레이션 벡터 (3,)
    fx,fy: focal length 각각 스칼라값
    cx,cy: principal point 각각 스칼라값
    k: 렌즈왜곡계수 (3,1)
    p: 렌즈왜곡계수 (2,1)
center: 센터 좌표 
scale : 스케일
box : 박스 좌표
"""




cams = [0,1,2,3]
all_cams = [0,1,2,3]

sample = dict()
anno3d = dict()


plt.ion()
fig = plt.figure(figsize=(12,12))
# for 0 in range(len(S_all_image_folder[0])):       # 18432개의 frame 

#     # if (0<len(S_all_image_folder[0]) )& (0 == 0):
# for c_idx, cam in enumerate(cams):
#     p2d = data[0]['joints_2d']
#     p3d = data[0]['joints_3d']

#     sample['cam' + str(data[0]['camera_id'])] = p2d
#     anno3d['cam' + str(data[0]['camera_id'])] = p3d


# poses_2d = sample
# poses_3d = anno3d  # 카메라0,1,2,3 순으로 

vis_2d_cam0 = data[0]['joints_2d']
vis_2d_cam1 = data[2356]['joints_2d']
# vis_2d_cam2 = poses_2d['cam2']
# vis_2d_cam3 = poses_2d['cam3']

# vis_3d_cam0 = poses_3d['cam0']
# R=data[0]['camera']['R']
# vis_3d_cam0_R = np.matmul(poses_3d['cam0'],R)
# vis_3d_cam1 = poses_3d['cam1']
# vis_3d_cam2 = poses_3d['cam2']
# vis_3d_cam3 = poses_3d['cam3']

# ax = fig.add_subplot('121', projection='3d', aspect='auto')
# show3Dpose(vis_3d_cam0, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))

# ax = fig.add_subplot('121', projection='3d', aspect='auto')
# show3Dpose(vis_3d_cam0_R, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))


# ax = fig.add_subplot('242', projection='3d', aspect='auto')
# show3Dpose(vis_3d_cam1, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))

# ax = fig.add_subplot('243', projection='3d', aspect='auto')
# show3Dpose(vis_3d_cam2, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))

# ax = fig.add_subplot('244', projection='3d', aspect='auto')
# show3Dpose(vis_3d_cam3, ax, data_type='h36m', radius=1500, lcolor='blue',angles=(20,-60))
# -----------------------------------------------------------------------------------------


ax = fig.add_subplot(121)
show2Dpose(vis_2d_cam0, ax, data_type='h36m', image_size=(1000,1002),img_path="images/"+data[0]['image'])

ax = fig.add_subplot(122)
show2Dpose(vis_2d_cam1, ax, data_type='h36m', image_size=(1000,1002),img_path="images/"+data[2356]['image'])

F=compute_fundamental(vis_2d_cam0,vis_2d_cam1)
line_x,line_y=plot_epipolar_line("images/"+data[2356]['image'],F,vis_2d_cam0[2])
ax.plot(line_x,line_y)

# ax = fig.add_subplot('247')
# show2Dpose(vis_2d_cam2, ax, data_type='h36m', image_size=(1000,1002))

# ax = fig.add_subplot('248')
# show2Dpose(vis_2d_cam3, ax, data_type='h36m', image_size=(1000,1002))

plt.draw()
plt.pause(100)
    # else:
    #     pass
    
