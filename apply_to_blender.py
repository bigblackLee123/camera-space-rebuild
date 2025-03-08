import json
import math
import os
import cv2  # 确保安装了 OpenCV 库
import numpy as np
import random
from typing import List, Dict, Tuple, Optional

def load_json_data(file_path):
    """从JSON文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_depth_image(depth_image_path):
    """加载深度图像"""
    if not os.path.exists(depth_image_path):
        print(f"深度图文件 {depth_image_path} 不存在")
        return None
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print("无法加载深度图像")
    return depth_image

def map_skeleton(frame_data, mmd_bones_data):
    """根据模型类型映射骨架并打印结果"""
    model_used = frame_data['model_used']
    frame_keypoints = frame_data['keypoints']
    frame_skeleton = frame_data['skeleton']
    
    print(f"使用的模型: {model_used}")
    
    # 硬编码深度图路径
    depth_image_path = "F:/ai_program_2/camera_estimate/workfile/paint_helper/output/seperate_frame/test_0109/full_body/frame_82/frame_82_depth.png"
    depth_image = load_depth_image(depth_image_path)
    
    if depth_image is not None:
        print("深度图像已成功加载")
    else:
        print("深度图像加载失败")
        return

    # 检查 frame_keypoints 的格式
    if not frame_keypoints or not isinstance(frame_keypoints, list):
        print("关键点数据格式错误")
        return
        
    # 检查 frame_skeleton 的格式
    if not frame_skeleton or not isinstance(frame_skeleton, list):
        print("骨架数据格式错误")
        return

    # 检查 mmd_bones_data 的格式
    if not isinstance(mmd_bones_data, dict):
        print("MMD骨骼数据格式错误")
        return

    if model_used == "full_body":
        mapping = {
            (0, 16): "首",
            (16, 10): "肩.L",
            (10, 11): "腕.L",
            (11, 12): "ひじ.L",
            (16, 13): "肩.R",
            (13, 14): "腕.R",
            (14, 15): "ひじ.R",
            (16, 1): "上半身2",
            (1, 8): "上半身",
            (8, 9): "下半身",
            (9, 2): None,
            (2, 4): "足.L",
            (4, 5): "ひざ.R",
            (9, 3): None,
            (3, 6): "足.R",
            (6, 7): "ひざ.L"
        }
    elif model_used == "complex_pose":
        mapping = {
            (0, 1): "首",
            (1, 4): "上半身2",
            (4, 5): "上半身",
            (1, 2): "肩.R",
            (1, 3): "肩.L",
            (2, 8): "腕.R",
            (8, 9): "ひじ.R",
            (3, 6): "腕.L",
            (6, 7): "ひじ.L",
            (2, 10): "胸上.L",
            (3, 11): "胸上.R",
            (4, 10): None,
            (4, 11): None
        }
    else:
        print(f"未知的模型类型: {model_used}")
        return
    
    # 存储有效的骨骼连接信息
    valid_connections = []
    
    # 显示映射结果
    for connection in frame_skeleton:
        try:
            connection_tuple = tuple(connection)  # 将列表转换为元组
            if connection_tuple in mapping:
                mapped_bone = mapping[connection_tuple]
                if mapped_bone and mapped_bone in mmd_bones_data:
                    # 计算连接线长度
                    start_idx, end_idx = connection
                    if (start_idx < len(frame_keypoints) and 
                        end_idx < len(frame_keypoints)):
                        
                        start_point = frame_keypoints[start_idx]
                        end_point = frame_keypoints[end_idx]
                        
                        # 确保点坐标是有效的
                        if (len(start_point) >= 2 and len(end_point) >= 2 and
                            0 <= int(start_point[1]) < depth_image.shape[0] and
                            0 <= int(start_point[0]) < depth_image.shape[1] and
                            0 <= int(end_point[1]) < depth_image.shape[0] and
                            0 <= int(end_point[0]) < depth_image.shape[1]):
                            
                            connection_length = ((end_point[0] - start_point[0]) ** 2 + 
                                              (end_point[1] - start_point[1]) ** 2) ** 0.5
                            bone_length = mmd_bones_data[mapped_bone]['length']
                            start_depth = depth_image[int(start_point[1]), int(start_point[0])]
                            end_depth = depth_image[int(end_point[1]), int(end_point[0])]
                            
                            print(f"连接 {connection} 映射到骨骼: {mapped_bone}")
                            print(f"  连接线长度: {connection_length:.2f} 像素")
                            print(f"  骨骼长度: {bone_length:.3f} 米")
                            print(f"  起点深度值: {start_depth}")
                            print(f"  终点深度值: {end_depth}")
                            
                            # 存储有效的连接信息
                            valid_connections.append({
                                'connection': connection,
                                'bone_name': mapped_bone,
                                'start_point': start_point,
                                'end_point': end_point,
                                'start_depth': start_depth,
                                'end_depth': end_depth,
                                'bone_length': bone_length,
                                'connection_length': connection_length
                            })
                    else:
                        print(f"连接 {connection} 的索引超出关键点列表范围")
                else:
                    print(f"连接 {connection} 映射的骨骼 {mapped_bone} 在MMD数据中未找到")
            else:
                print(f"连接 {connection} 没有映射到任何骨骼")
                
        except Exception as e:
            print(f"处理连接 {connection} 时出错: {str(e)}")
            
    return valid_connections

def calculate_camera_intrinsics(image_width: int, image_height: int) -> np.ndarray:
    """
    计算相机内参矩阵 (使用28mm焦距)
    """
    # 50mm 焦距，35mm全画幅传感器 (36mm x 24mm)
    focal_length_mm = 28
    sensor_width_mm = 36
    
    # 计算像素单位的焦距
    focal_length_px = (focal_length_mm * image_width) / sensor_width_mm
    cx = image_width / 2
    cy = image_height / 2
    
    # 构建内参矩阵
    K = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ])
    
    print(f"相机内参：")
    print(f"焦距（毫米）: {focal_length_mm}")
    print(f"焦距（像素）: {focal_length_px}")
    print(f"主点坐标: ({cx}, {cy})")
    print(f"内参矩阵K:\n{K}")
    
    return K

def calculate_3d_points(valid_connections: List[Dict], K: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    根据有效连接信息计算3D点对
    """
    points_3d = []
    points_2d = []
    K_inv = np.linalg.inv(K)
    
    for conn in valid_connections:
        start_point = np.array(conn['start_point'])
        end_point = np.array(conn['end_point'])
        start_depth = conn['start_depth']
        end_depth = conn['end_depth']
        bone_length = conn['bone_length']  # 实际骨骼长度（米）
        
        # 计算方向向量
        start_dir = K_inv @ np.array([start_point[0], start_point[1], 1])
        end_dir = K_inv @ np.array([end_point[0], end_point[1], 1])
        
        # 归一化方向向量
        start_dir = start_dir / np.linalg.norm(start_dir)
        end_dir = end_dir / np.linalg.norm(end_dir)
        
        # 根据深度比例和实际骨骼长度计算scale因子
        depth_ratio = end_depth / start_depth
        # 解二次方程求scale
        a = np.sum((start_dir - depth_ratio * end_dir) ** 2)
        b = 0
        c = -bone_length ** 2
        scale = np.sqrt(-c/a)
        
        # 计算3D点
        point1_3d = scale * start_dir
        point2_3d = scale * depth_ratio * end_dir
        
        points_3d.extend([point1_3d, point2_3d])
        points_2d.extend([start_point[:2], end_point[:2]])
    
    return np.array(points_3d), np.array(points_2d)

def rotationMatrixToEulerAngles(R):
    """
    将3x3旋转矩阵转换为欧拉角（XYZ顺序）
    返回：(rx, ry, rz) 弧度制
    """
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def generate_blender_camera_script(K, R, tvec, output_path):
    """生成Blender相机设置脚本
    Args:
        K: 相机内参矩阵
        R: 旋转矩阵
        tvec: 平移向量（相机位置）
        output_path: 输出脚本路径
    """
    # 1. 直接使用tvec作为相机位置（记得转换回米）
    MM_TO_BLENDER = 0.001  # 1毫米 = 0.001米
    camera_position = tvec.flatten() * MM_TO_BLENDER
    
    # 2. 从旋转矩阵计算欧拉角
    euler_angles = rotationMatrixToEulerAngles(R)
    
    # 3. 生成Blender脚本
    script_content = f"""
import bpy
import math

# 设置相机位置
bpy.data.objects['Camera'].location = ({camera_position[0]}, {camera_position[1]}, {camera_position[2]})

# 设置相机旋转（欧拉角，弧度制）
bpy.data.objects['Camera'].rotation_euler = ({euler_angles[0]}, {euler_angles[1]}, {euler_angles[2]})

# 设置相机内参
bpy.data.cameras['Camera'].lens = {K[0,0]}  # 焦距
"""
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write(script_content)

def find_center_anchor(frame_data, mmd_bones_data, depth_image):
    """
    寻找距离画面中心最近的有效骨骼连接线作为锚点
    
    Args:
        frame_data: 包含keypoints和skeleton的帧数据
        mmd_bones_data: MMD骨骼数据
        depth_image: 深度图像
    
    Returns:
        anchor_info: 包含锚点所有必要信息的字典，如果没找到则返回None
    """
    # 获取图像中心点
    height, width = depth_image.shape[:2]
    center_x = width / 2
    center_y = height / 2
    
    # 获取模型类型和对应的骨骼映射
    model_used = frame_data['model_used']
    if model_used == "full_body":
        mapping = {
            (0, 16): "首",
            (16, 10): "肩.L",
            (10, 11): "腕.L",
            (11, 12): "ひじ.L",
            (16, 13): "肩.R",
            (13, 14): "腕.R",
            (14, 15): "ひじ.R",
            (16, 1): "上半身2",
            (1, 8): "上半身",
            (8, 9): "下半身",
            (9, 2): None,
            (2, 4): "足.L",
            (4, 5): "ひざ.R",
            (9, 3): None,
            (3, 6): "足.R",
            (6, 7): "ひざ.L"
        }
    elif model_used == "complex_pose":
        mapping = {
            (0, 1): "首",
            (1, 4): "上半身2",
            (4, 5): "上半身",
            (1, 2): "肩.R",
            (1, 3): "肩.L",
            (2, 8): "腕.R",
            (8, 9): "ひじ.R",
            (3, 6): "腕.L",
            (6, 7): "ひじ.L",
            (2, 10): "胸上.L",
            (3, 11): "胸上.R",
            (4, 10): None,
            (4, 11): None
        }
    else:
        print(f"未知的模型类型: {model_used}")
        return
    


def get_reference_points():
    """
    定义需要的参考点及其对应关系
    返回：骨骼连接与关键点索引的映射
    """
    reference_mapping = {
        '下半身': {
            'start_idx': 9,    # spine base
            'end_idx': 8,      # spine1
            'bone_name': '下半身'
        },
        '足D.L': {
            'start_idx': 2,    # left hip
            'bone_name': '足D.L'
        },
        '足D.R': {
            'start_idx': 3,    # right hip
            'bone_name': '足D.R'
        },
        '上半身2': {
            'start_idx': 1,    # spine2
            'end_idx': 16,     # neck
            'bone_name': '上半身2'
        }
    }
    return reference_mapping

def collect_3d_2d_points(mmd_bones_data, keypoints_2d, reference_mapping):
    points_3d = []
    points_2d = []
    
    # 添加单位转换：从米转换为毫米
    BLENDER_TO_MM = 1000.0  # 1米 = 1000毫米
    
    # 1. 添加下半身骨骼的起点和终点
    bone_info = reference_mapping['下半身']
    if bone_info['bone_name'] in mmd_bones_data:
        # 添加起点（转换为毫米）
        start_3d = np.array(mmd_bones_data[bone_info['bone_name']]['head_position']) * BLENDER_TO_MM
        start_2d = np.array(keypoints_2d[bone_info['start_idx']])
        points_3d.append(start_3d)
        points_2d.append(start_2d)
        
        # 添加终点（转换为毫米）
        end_3d = np.array(mmd_bones_data[bone_info['bone_name']]['tail_position']) * BLENDER_TO_MM
        end_2d = np.array(keypoints_2d[bone_info['end_idx']])
        points_3d.append(end_3d)
        points_2d.append(end_2d)
        
        print(f"下半身骨骼:")
        print(f"  起点3D (mm): {start_3d}")
        print(f"  起点2D: {start_2d}")
        print(f"  终点3D (mm): {end_3d}")
        print(f"  终点2D: {end_2d}")
    
    # 2. 添加左右足的起始点
    for leg_name in ['足D.L', '足D.R']:
        bone_info = reference_mapping[leg_name]
        if bone_info['bone_name'] in mmd_bones_data:
            point_3d = np.array(mmd_bones_data[bone_info['bone_name']]['head_position']) * BLENDER_TO_MM
            point_2d = np.array(keypoints_2d[bone_info['start_idx']])
            points_3d.append(point_3d)
            points_2d.append(point_2d)
            
            print(f"{leg_name}:")
            print(f"  起点3D (mm): {point_3d}")
            print(f"  起点2D: {point_2d}")
    
    # 3. 添加上半身2的起点和终点
    bone_info = reference_mapping['上半身2']
    if bone_info['bone_name'] in mmd_bones_data:
        start_3d = np.array(mmd_bones_data[bone_info['bone_name']]['head_position']) * BLENDER_TO_MM
        start_2d = np.array(keypoints_2d[bone_info['start_idx']])
        points_3d.append(start_3d)
        points_2d.append(start_2d)
        
        end_3d = np.array(mmd_bones_data[bone_info['bone_name']]['tail_position']) * BLENDER_TO_MM
        end_2d = np.array(keypoints_2d[bone_info['end_idx']])
        points_3d.append(end_3d)
        points_2d.append(end_2d)
        
        print(f"上半身2骨骼:")
        print(f"  起点3D (mm): {start_3d}")
        print(f"  起点2D: {start_2d}")
        print(f"  终点3D (mm): {end_3d}")
        print(f"  终点2D: {end_2d}")
    
    return np.array(points_3d, dtype=np.float32), np.array(points_2d, dtype=np.float32)

def visualize_projection_points(image_size, points_2d, projected_points, output_path="projection_visualization.png"):
    """
    可视化投影点和实际2D点
    
    Args:
        image_size: (height, width) 图像尺寸
        points_2d: 实际2D点坐标
        projected_points: 投影后的2D点坐标
        output_path: 输出图像路径
    """
    # 创建白色背景图像
    height, width = image_size
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 将投影点的形状调整为(N,2)
    proj_points = projected_points.reshape(-1, 2)
    
    # 绘制点和连线
    for i in range(len(points_2d)):
        # 转换为整数坐标
        actual = tuple(map(int, points_2d[i]))
        projected = tuple(map(int, proj_points[i]))
        
        # 绘制实际点（蓝色）
        cv2.circle(visualization, actual, 5, (255, 0, 0), -1)
        # 绘制投影点（红色）
        cv2.circle(visualization, projected, 5, (0, 0, 255), -1)
        # 绘制连线（绿色）
        cv2.line(visualization, actual, projected, (0, 255, 0), 1)
        
        # 添加点的索引标签
        cv2.putText(visualization, f'{i}', actual, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 打印点的坐标
        print(f"点 {i}:")
        print(f"  实际坐标: ({actual[0]}, {actual[1]})")
        print(f"  投影坐标: ({projected[0]}, {projected[1]})")
        print(f"  误差: {np.linalg.norm(points_2d[i] - proj_points[i]):.2f}像素")
    
    # 添加图例
    legend_y = 30
    cv2.putText(visualization, "蓝色: 实际点", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(visualization, "红色: 投影点", (10, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(visualization, "绿线: 误差", (10, legend_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存图像
    cv2.imwrite(output_path, visualization)
    print(f"可视化结果已保存到: {output_path}")

def blender_to_opencv_coords(points_3d):
    """
    Blender到OpenCV的坐标系转换
    Blender: 右手坐标系 (X右, Y内, Z上)
    OpenCV: 右手坐标系 (X右, Y下, Z前)
    """
    points_cv = points_3d.copy()
    # Y轴和Z轴交换，Y轴取反
    points_cv[..., 1], points_cv[..., 2] = -points_3d[..., 2], points_3d[..., 1]
    return points_cv

def opencv_to_blender_coords(points_cv):
    """
    OpenCV到Blender的坐标系转换
    """
    points_blender = points_cv.copy()
    # Y轴和Z轴交换回来，Z轴取反
    points_blender[..., 1], points_blender[..., 2] = points_cv[..., 2], -points_cv[..., 1]
    return points_blender

def solve_camera_pose_ransac(points_3d, points_2d, K, ransac_iterations=1000, ransac_threshold=8.0):
    """
    使用RANSAC方法求解相机位姿
    
    Args:
        points_3d: 3D点坐标 (N,3)
        points_2d: 2D点坐标 (N,2)
        K: 相机内参矩阵
        ransac_iterations: RANSAC迭代次数
        ransac_threshold: 内点判断阈值（像素）
    
    Returns:
        best_R: 最优旋转矩阵
        best_t: 最优平移向量
        inliers: 内点索引
    """
    best_error = float('inf')
    best_R = None
    best_t = None
    best_inliers = None
    n_points = points_3d.shape[0]
    
    # 确保至少有6个点（PnP最小点数要求）
    min_points = 6
    if n_points < min_points:
        raise ValueError(f"需要至少{min_points}个点，当前只有{n_points}个点")
    
    for _ in range(ransac_iterations):
        # 1. 随机选择6个点
        indices = np.random.choice(n_points, min_points, replace=False)
        pts_3d = points_3d[indices]
        pts_2d = points_2d[indices]
        
        try:
            # 2. 使用选中的点计算位姿
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                continue
                
            # 3. 计算所有点的重投影误差
            projected_points, _ = cv2.projectPoints(
                points_3d, rvec, tvec, K, None
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # 4. 计算每个点的重投影误差
            errors = np.linalg.norm(points_2d - projected_points, axis=1)
            
            # 5. 找出内点
            inliers = errors < ransac_threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count >= 6:
                # 6. 使用所有内点重新计算位姿
                success, rvec_refined, tvec_refined = cv2.solvePnP(
                    points_3d[inliers],
                    points_2d[inliers],
                    K, None,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # 计算使用所有内点的平均误差
                    projected_points_refined, _ = cv2.projectPoints(
                        points_3d, rvec_refined, tvec_refined, K, None
                    )
                    errors_refined = np.linalg.norm(
                        points_2d - projected_points_refined.reshape(-1, 2),
                        axis=1
                    )
                    mean_error = np.mean(errors_refined[inliers])
                    
                    # 更新最佳结果
                    if mean_error < best_error:
                        best_error = mean_error
                        R_mat, _ = cv2.Rodrigues(rvec_refined)
                        best_R = R_mat
                        best_t = tvec_refined
                        best_inliers = inliers
                        
        except cv2.error:
            continue
            
    if best_R is None:
        raise RuntimeError("RANSAC未能找到好的解")
        
    return best_R, best_t, best_inliers

def main():
    # 硬编码文件路径
    frame_file_path = "F:/ai_program_2/camera_estimate/workfile/paint_helper/output/seperate_frame/test_0109/keypoints/keypoints_frame_82.json"
    mmd_bones_file_path = "F:/ai_program_2/camera_estimate/workfile/paint_helper/output/mmd_bones_data.json"
    depth_image_path = "F:/ai_program_2/camera_estimate/workfile/paint_helper/output/seperate_frame/test_0109/full_body/frame_82/frame_82_depth.png"

    # 检查文件是否存在
    if not os.path.exists(frame_file_path):
        print(f"文件 {frame_file_path} 不存在")
        return
    if not os.path.exists(mmd_bones_file_path):
        print(f"文件 {mmd_bones_file_path} 不存在")
        return
    
    try:
        # 1. 加载深度图像
        depth_image = load_depth_image(depth_image_path)
        if depth_image is None:
            print("无法加载深度图像")
            return
            
        # 2. 加载帧中的动作数据
        frame_data = load_json_data(frame_file_path)
        keypoints_2d = frame_data['keypoints']  # 这是一个列表，每个元素是[x,y]坐标
        
        print("关键点数据示例：")
        for i, kp in enumerate(keypoints_2d):
            # kp 是一个包含两个元素的列表 [x, y]
            print(f"关键点 {i}: ({kp[0]}, {kp[1]})")
        
        # 3. 加载模型骨骼数据
        mmd_bones_data = load_json_data(mmd_bones_file_path)
        
        # 4. 获取参考点映射
        reference_mapping = get_reference_points()
        
        # 5. 收集3D-2D对应点
        points_3d, points_2d = collect_3d_2d_points(
            mmd_bones_data, 
            keypoints_2d,
            reference_mapping
        )
        
        # 转换到OpenCV坐标系
        points_3d_cv = blender_to_opencv_coords(points_3d)
        
        # 6. 计算相机内参
        height, width = depth_image.shape[:2]
        K = calculate_camera_intrinsics(width, height)
        
        # 7. 使用RANSAC求解相机位姿
        try:
            R, tvec, inliers = solve_camera_pose_ransac(
                points_3d_cv,
                points_2d,
                K,
                ransac_iterations=1000,
                ransac_threshold=8.0
            )
            
            # 转换回Blender坐标系
            tvec_blender = opencv_to_blender_coords(tvec.reshape(1, 3)).flatten()
            R_blender = opencv_to_blender_coords(R)
            
            print(f"\n计算结果：")
            print(f"内点数量: {np.sum(inliers)}/{len(inliers)}")
            print(f"相机位置（Blender）：{tvec_blender}")
            print(f"相机旋转矩阵（Blender）：\n{R_blender}")
            
            # 验证结果：计算重投影误差
            projected_points, _ = cv2.projectPoints(
                points_3d_cv, 
                cv2.Rodrigues(R)[0], 
                tvec, 
                K, 
                None
            )
            
            error = np.mean(np.linalg.norm(
                points_2d[inliers] - projected_points.reshape(-1, 2)[inliers],
                axis=1
            ))
            print(f"重投影误差（仅内点）: {error:.3f} 像素")
            
            # 可视化投影结果
            visualize_projection_points(
                image_size=(height, width),
                points_2d=points_2d,
                projected_points=projected_points,
                inliers=inliers,  # 添加内点信息
                output_path="projection_visualization.png"
            )
            
            # 生成Blender相机脚本
            output_script_path = "camera_setup.py"
            generate_blender_camera_script(K, R_blender, tvec_blender, output_script_path)
            
        except Exception as e:
            print(f"相机位姿求解失败: {str(e)}")
            
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()