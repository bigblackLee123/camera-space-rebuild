import cv2
import numpy as np
import json
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
import traceback
import time
import os
import bpy
from mathutils import Vector
import math
from mathutils import Matrix, Euler
import subprocess
import sys

# 获取当前脚本所在目录
current_dir = Path(__file__).parent

class SketchAnnotator:
    def __init__(self):
        self.image = None
        self.points = []
        self.lines = []
        self.current_point = None
        self.window_name = "草图标注工具"
        self.is_running = True
        self.enable_manual_annotation = False
        self.window_visible = True  # 添加窗口状态标志
        
    def load_image(self, image_path):
        """从文件加载图像"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("无法加载图像")
        self.display_image = self.image.copy()
        
    def load_frame(self, frame):
        """直接加载帧数据"""
        if frame is not None:
            self.image = frame.copy()
            self.display_image = self.image.copy()
        else:
            raise ValueError("无效的帧数据")        

    def start_annotation(self):
        """开始标注过程"""
        try:
            cv2.namedWindow(self.window_name)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            while self.is_running:
                # 检查窗口是否被关闭
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 0:
                        self.window_visible = False
                        break
                except:
                    self.window_visible = False
                    break
                    
                if not self.window_visible:
                    break
                    
                cv2.imshow(self.window_name, self.display_image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q或ESC退出
                    break
                    
            self.is_running = False
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
            
        except Exception as e:
            self.is_running = False
            print(f"标注过程出错: {str(e)}")
            traceback.print_exc()
        
    

    def close_window(self):
        """关闭窗口并清理资源"""
        self.is_running = False
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
    def save_annotations(self, output_path):
        """保存标注数据为COCO格式"""
        annotations = {
            "keypoints": self.points,
            "skeleton": self.lines
        }
        
        with open(output_path, 'w') as f:
            json.dump(annotations, f)
            
    def export_keypoints(self, base_dir, video_name, frame_number, model_type):
        """导出关键点数据到指定目录"""
        try:
            # 创建目标目录
            keypoints_dir = os.path.join(base_dir, video_name, "keypoints")
            os.makedirs(keypoints_dir, exist_ok=True)
            
            # 动态生成输出文件路径，包含帧号
            output_filename = f"keypoints_frame_{frame_number}.json"
            output_path = os.path.join(keypoints_dir, output_filename)
            
            # 保存标注数据
            annotations = {
                "model_used": model_type,  # 添加模型类型
                "keypoints": self.points,
                "skeleton": self.lines
            }
            
            with open(output_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            print(f"关键点数据已导出到 {output_path}")
        
        except Exception as e:
            print(f"导出关键点数据时出错: {str(e)}")

class COCOKeypointProcessor:
    def __init__(self):
        self.keypoints = None
        self.skeleton = None
        self.offset_x = 0
        self.offset_y = 0
        self.selected_point = None
        self.drag_threshold = 5
        self.is_dragging = False
        self.current_image = None
        self.window_visible = True

    def set_offset(self, offset_x, offset_y):
        """设置坐标偏移量"""
        self.offset_x = offset_x
        self.offset_y = offset_y
        #logger.info(f"设置偏移量: x={offset_x}, y={offset_y}")

    def find_nearest_point(self, x, y):
        """找到最近的关键点"""
        if not self.keypoints:
            return None
            
        # 计算缩放后的关键点位置
        height, width = self.current_image.shape[:2]
        scale_x = width / 1280
        scale_y = height / 720
        
        adjusted_keypoints = [
            (int(kp[0] * scale_x + self.offset_x), 
             int(kp[1] * scale_y + self.offset_y))
            for kp in self.keypoints
        ]
        
        # 找到最近的点
        min_dist = float('inf')
        nearest_idx = None
        
        for i, point in enumerate(adjusted_keypoints):
            dist = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
            if dist < min_dist and dist < self.drag_threshold * 2:  # 增加选择范围
                min_dist = dist
                nearest_idx = i
                
        return nearest_idx
        
    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件"""
        try:
            # 基础检查
            if not self._check_window_state(param):
                return
            
            if not self.keypoints:
                return
            
            # 获取并保存图像引用
            image = param.get('image')
            if image is None:
                return
            self.current_image = image

            window_name = param.get('window_name')
            
            # 处理鼠标事件
            if event == cv2.EVENT_LBUTTONDOWN:
                # 选择最近的关键点
                self.selected_point = self.find_nearest_point(x, y)
                if self.selected_point is not None:
                    self.is_dragging = True
                    print(f"开始拖动关键点 {self.selected_point}")
                    
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:  # 简化条件判断
                # 拖动选中的关键点
                if self.selected_point is not None:
                    # 计算缩放比例
                    height, width = image.shape[:2]
                    scale_x = width / 1280
                    scale_y = height / 720
                    
                    # 更新关键点位置
                    new_x = (x - self.offset_x) / scale_x
                    new_y = (y - self.offset_y) / scale_y
                    
                    if isinstance(self.keypoints, list) and self.selected_point < len(self.keypoints):
                        self.keypoints[self.selected_point] = (new_x, new_y)
                
                # 重新绘制
                    if hasattr(self, 'original_image'):
                        result = self.draw_on_sketch(self.original_image, log=False)
                    else:
                        result = self.draw_on_sketch(image, log=False)
                    cv2.imshow(window_name, result)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                # 释放选中的关键点
                if self.selected_point is not None and self.is_dragging:
                    print(f"完成拖动关键点 {self.selected_point}")
                    self.is_dragging = False
                    self.selected_point = None
                    # 最后一次绘制
                    #result = self.draw_on_sketch(image, log=True)
                    #cv2.imshow(window_name, result)
                    
        except Exception as e:
            print(f"鼠标事件处理出错: {str(e)}")
            traceback.print_exc()

    def _check_window_state(self, param):
        """检查窗口状态"""
        if not self.window_visible:
            return False
        
        window_name = param.get('window_name')
        if not window_name:
            return False
        
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                self.window_visible = False
                return False
        except:
            self.window_visible = False
            return False
        
        return True

    def _update_display(self, window_name, image, log=False):
        """更新显示"""
        if window_name and image is not None:
            result = self.draw_on_sketch(image, log=log)
            cv2.imshow(window_name, result)

    def setup_complex_pose_skeleton(self):
        """设置复杂姿势的骨架连接"""
        self.skeleton = [
            [0, 1],   # head -> neck
            [1, 4],   # neck -> spine2
            [4, 5],   # spine2 -> spine1
            [1, 2],   # neck -> shoulder_R
            [1, 3],   # neck -> shoulder_L
            [2, 8],   # shoulder_R -> joint_R
            [8, 9],   # joint_R -> hand_R
            [3, 6],   # shoulder_L -> joint_L
            [6, 7],   # joint_L -> hand_L
            [2, 10],  # shoulder_R -> ribs_R
            [3, 11],  # shoulder_L -> ribs_L
            [4, 10],  # spine2 -> ribs_R
            [4, 11],  # spine2 -> ribs_L
        ]
        
    def setup_full_body_skeleton(self):
        """设置全身姿势的骨架连接"""
        self.skeleton = [
            [0, 16],  # head -> neck
            [16, 10], # neck -> spine2
            [10, 11],  # spine2 -> spine1
            [11, 12],   # spine1 -> spine
            [16, 13], # neck -> clavicle_L
            [13, 14], # clavicle_L -> joint_L
            [14, 15], # joint_L -> hand_L
            [16, 1], # neck -> clavicle_R
            [1, 8], # clavicle_R -> joint_R
            [8, 9], # joint_R -> hand_R
            [9, 2],   # spine -> thigh_L
            [2, 4],   # thigh_L -> calf_L
            [4, 5],   # calf_L -> foot_L
            [9, 3],   # spine -> thigh_R
            [3, 6],   # thigh_R -> calf_R
            [6, 7],   # calf_R -> foot_R
        ]

    def load_complex_pose_data(self, results):
        """加载复杂姿态数据"""
        print("加载前的关键点:", self.keypoints)  # 调试：查看原有数据
        self.keypoints = results.get('keypoints', [])  # 从结果中获取关键点列表
        self.skeleton = results.get('skeleton', [])    # 从结果中获取骨架连接信息
        print("加载后的关键点:", self.keypoints)  # 调试：确认新数据已加载

    def load_full_body_data(self, results):
        """加载全身姿势数据"""
        try:
            if not isinstance(results, dict):
                raise ValueError("输入数据必须是字典类型")
                
            # 正确的数据访问路径
            predictions = results.get('predictions', [])
            if not predictions:
                print("DEBUG: results =", results)  # 调试输出
                raise ValueError("无法找到预测数据")
                
            first_prediction = predictions[0].get('predictions', [])
            if not first_prediction:
                raise ValueError("无法找到第一层预测数据")
                
            keypoints_data = first_prediction[0].get('keypoints', [])
            if not keypoints_data:
                raise ValueError("无法找到关键点数据")
                
            # 转换为坐标列表
            self.keypoints = [(kp["x"], kp["y"]) for kp in keypoints_data]
            
            # 设置骨架连接
            self.setup_full_body_skeleton()
            print(f"成功加载 {len(self.keypoints)} 个全身姿势关键点")
            return True
            
        except Exception as e:
            print(f"加载全身姿势数据时出错: {str(e)}")
            print("输入数据:", results)  # 添加调试输出
            traceback.print_exc()
            return False

        
    def draw_on_sketch(self, sketch_image, log=True):
        """在草图上绘制关键点和骨架"""
        try:
            # 基本检查
            if not self.window_visible or self.keypoints is None:
                return sketch_image
            
            # 保存原始图像（仅第一次）
            if not hasattr(self, 'original_image'):
                self.original_image = sketch_image.copy()
            
            # 创建新的绘制画布
            result = self.original_image.copy()
            height, width = result.shape[:2]
            scale_x = width / 1280
            scale_y = height / 720
            
            # 处理关键点坐标
            adjusted_keypoints = []
            for i, point in enumerate(self.keypoints):
                if point is not None:
                    x, y = point
                    # 应用缩放和偏移
                    adj_x = int(x * scale_x + self.offset_x)
                    adj_y = int(y * scale_y + self.offset_y)
                    adjusted_keypoints.append((adj_x, adj_y))
            
            # 定义深色调配色方案
            colors = {
                'deep_teal': (128, 128, 32),      # 深青色 - 中心线
                'ocean_blue': (128, 64, 32),      # 海蓝色 - 肩膀连接
                'forest_green': (64, 128, 32),    # 森林绿 - 手臂
                'deep_purple': (128, 32, 64),     # 深紫色 - 躯干
                'point_color': (140, 120, 40),    # 关键点颜色
                'outline_color': (48, 96, 96)     # 关键点轮廓
            }
            
            # 为不同部分定义颜色
            skeleton_colors = [
                'deep_teal',    # head -> neck
                'deep_teal',    # neck -> spine2
                'deep_teal',    # spine2 -> spine1
                'deep_teal',    # spine1 -> spine
                'ocean_blue',   # neck -> clavicle_L
                'forest_green', # clavicle_L -> joint_L
                'forest_green', # joint_L -> hand_L
                'ocean_blue',   # neck -> clavicle_R
                'forest_green', # clavicle_R -> joint_R
                'forest_green', # joint_R -> hand_R
                'deep_purple',  # spine -> thigh_L
                'deep_purple',  # thigh_L -> calf_L
                'deep_purple',  # calf_L -> foot_L
                'deep_purple',  # spine -> thigh_R
                'deep_purple',  # thigh_R -> calf_R
                'deep_purple'   # calf_R -> foot_R
            ]
            
            # 绘制骨架连线 - 更细的线条
            if self.skeleton:
                for (line, color) in zip(self.skeleton, skeleton_colors):
                    if (0 <= line[0] < len(adjusted_keypoints) and 
                        0 <= line[1] < len(adjusted_keypoints)):
                        pt1 = adjusted_keypoints[line[0]]
                        pt2 = adjusted_keypoints[line[1]]
                        # 主线条更细
                        cv2.line(result, pt1, pt2, colors[color], 1, cv2.LINE_AA)
            
            # 绘制关键点 - 更小的点
            for point in adjusted_keypoints:
                # 外圈 - 更小的半径
                cv2.circle(result, point, 4, colors['outline_color'], 1, cv2.LINE_AA)
                # 内圈 - 更小的实心点
                cv2.circle(result, point, 3, colors['point_color'], -1, cv2.LINE_AA)
                # 高光效果 - 更小的高光点
                cv2.circle(result, point, 1, 
                          tuple(c + 40 for c in colors['point_color']), -1, cv2.LINE_AA)
            
            if log:
                print(f"绘制了 {len(adjusted_keypoints)} 个关键点")
            return result
            
        except Exception as e:
            print(f"绘制关键点时出错: {str(e)}")
            traceback.print_exc()
            return sketch_image

    def load_complex_pose_data(self, results):
        """加载复杂姿势数据"""
        try:
            if not isinstance(results, dict):
                raise ValueError("输入数据必须是字典类型")
                
            # 正确的数据访问路径
            predictions = results.get('predictions', [])
            if not predictions:
                print("DEBUG: results =", results)  # 调试输出
                raise ValueError("无法找到预测数据")
                
            first_prediction = predictions[0].get('predictions', [])
            if not first_prediction:
                raise ValueError("无法找到第一层预测数据")
                
            keypoints_data = first_prediction[0].get('keypoints', [])
            if not keypoints_data:
                raise ValueError("无法找到关键点数据")
                
            # 转换为坐标列表
            self.keypoints = [(kp["x"], kp["y"]) for kp in keypoints_data]
            
            # 设置骨架连接
            self.setup_complex_pose_skeleton()
            print(f"成功加载 {len(self.keypoints)} 个复杂姿势关键点")
            return True
            
        except Exception as e:
            print(f"加载复杂姿势数据时出错: {str(e)}")
            print("输入数据:", results)  # 添加调试输出
            traceback.print_exc()
            return False
            
    def load_full_body_data(self, results):
        """加载全身姿势数据"""
        try:
            if not isinstance(results, dict):
                raise ValueError("输入数据必须是字典类型")
                
            # 正确的数据访问路径
            predictions = results.get('predictions', [])
            if not predictions:
                print("DEBUG: results =", results)  # 调试输出
                raise ValueError("无法找到预测数据")
                
            first_prediction = predictions[0].get('predictions', [])
            if not first_prediction:
                raise ValueError("无法找到第一层预测数据")
                
            keypoints_data = first_prediction[0].get('keypoints', [])
            if not keypoints_data:
                raise ValueError("无法找到关键点数据")
                
            # 转换为坐标列表
            self.keypoints = [(kp["x"], kp["y"]) for kp in keypoints_data]
            
            # 设置骨架连接
            self.setup_full_body_skeleton()
            print(f"成功加载 {len(self.keypoints)} 个全身姿势关键点")
            return True
            
        except Exception as e:
            print(f"加载全身姿势数据时出错: {str(e)}")
            print("输入数据:", results)  # 添加调试输出
            traceback.print_exc()
            return False


class FullBodyKeypointProcessor:
    def __init__(self):
        self.keypoints = None
        self.skeleton = None
        self.offset_x = 0
        self.offset_y = 0
        self.selected_point = None
        self.drag_threshold = 5

    def load_full_body_format(self, txt_path):
        """加载全身姿态数据"""
        try:
            with open(txt_path, 'r') as f:
                data = json.load(f)
            
            if not data or "predictions" not in data:
                print("数据格式错误或为空")
                return False
            
            # 提取关键点数据
            keypoints_data = data["predictions"]["results"]["predictions"][0]["predictions"][0]["keypoints"]
            self.keypoints = [(kp["x"], kp["y"]) for kp in keypoints_data]
            
            # 定义全身骨架连接
            self.skeleton = [
                # 头部和躯干
                [0, 16],  # head -> neck
                [16, 10], # neck -> spine2
                [10, 11],  # spine2 -> spine1
                [11, 12],   # spine1 -> spine
                
                # 左臂
                [16, 10], # neck -> clavicle_L
                [10, 11], # clavicle_L -> joint_L
                [11, 12], # joint_L -> hand_L
                
                # 右臂
                [16, 13], # neck -> clavicle_R
                [13, 14], # clavicle_R -> joint_R
                [14, 15], # joint_R -> hand_R
                
                # 左腿
                [1, 2],   # spine -> thigh_L
                [2, 4],   # thigh_L -> calf_L
                [4, 5],   # calf_L -> foot_L
                
                # 右腿
                [1, 3],   # spine -> thigh_R
                [3, 6],   # thigh_R -> calf_R
                [6, 7],   # calf_R -> foot_R
            ]
            
            print(f"成功加载 {len(self.keypoints)} 个全身关键点")
            return True
            
        except Exception as e:
            print(f"加载全身姿态数据时出错: {str(e)}")
            return False

def main():
        # 设置文件路径
    sketch_path = "F:/ai_program_2/camera_estimate/video_cut/valid/e8d836dd4a9cca5dfd716a68964a5bc.png"
    coco_path = "F:/ai_program_2/camera_estimate/video_cut/valid/coco.txt"
    
    try:
        # 加载图像
        print("正在加载草图...")
        image = cv2.imread(sketch_path)
        #if image is None:
        #    raise ValueError("无法加载图像")
            
        # 调整图像大小
        #max_size = 800
        #height, width = image.shape[:2]
        #if width > max_size or height > max_size:
        #    if width > height:
        #        new_width = max_size
        #        new_height = int(height * (max_size / width))
        #    else:
        #        new_height = max_size
        #        new_width = int(width * (max_size / height))
        #    image = cv2.resize(image, (new_width, new_height))
        
        # 处理COCO格式数据
        print("\n正在加载COCO数据...")
        processor = COCOKeypointProcessor()
        if not processor.load_coco_format(coco_path):
            raise ValueError("无法加载COCO数据")
            
        # 初始化深度估计器
        #print("\n初始化深度估计器...")
        #depth_estimator = SketchDepthEstimator()
        
        # 创建窗口
        window_name = "骨架调整"
        #depth_window = "深度估计"
        
        # 确保窗口正确创建
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)
        
        # 显示初始图像
        cv2.imshow(window_name, image)
        cv2.waitKey(100)  # 等待窗口创建完成
        
        # 创建轨迹条
        cv2.createTrackbar('X偏移', window_name, 400, 800, lambda x: None)
        cv2.createTrackbar('Y偏移', window_name, 300, 600, lambda x: None)
        
        # 设置鼠标回调
        #cv2.setMouseCallback(window_name, 
        #                       processor.mouse_callback, 
        #                       {'image': image, 'window_name': window_name})
        
        print("\n程序已启动:")
        print("- 使用滑动条调整位置")
        print("- 按'd'键生成深度图")
        print("- 按's'键保存结果")
        print("- 按'q'键退出程序")
        
        while True:
            try:
                # 获取当前偏移值
                offset_x = cv2.getTrackbarPos('X偏移', window_name) - 400
                offset_y = cv2.getTrackbarPos('Y偏移', window_name) - 300
                
                # 设置偏移
                processor.set_offset(offset_x, offset_y)
                
                # 绘制结果
                #result = processor.draw_on_sketch(image)
                
                # 显示图像
                #cv2.imshow(window_name, result)
                
                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 按q退出
                    break
                elif key == ord('s'):  # 按s保存
                    # 保存调整后的键点数据
                    with open(coco_path, 'r') as f:
                        data = json.load(f)
                    
                    # 更新关键点坐标
                    for i, (x, y) in enumerate(processor.keypoints):
                        data['predictions'][0]['keypoints'][i]['x'] = x
                        data['predictions'][0]['keypoints'][i]['y'] = y
                    
                    # 保存到新文件
                    save_path = 'adjusted_keypoints.txt'
                    with open(save_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # 保存图像
                    cv2.imwrite('adjusted_skeleton.png', result)
                    print(f"结果已保存至 {save_path} 和 adjusted_skeleton.png")
               
                        
            except cv2.error as e:
                print(f"OpenCV错误: {str(e)}")
                break
                
        cv2.destroyAllWindows()
        print("\n程序已退出")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
