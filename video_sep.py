import sys
import torch
import onnxruntime as ort
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import roboflow
import os

def check_versions():
    """检查环境版本信息"""
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # CUDA相关信息
    print("\nCUDA信息:")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDNN版本: {torch.backends.cudnn.version()}")
    
    # ONNX Runtime信息
    print("\nONNX Runtime信息:")
    print(f"ONNX Runtime版本: {ort.__version__}")
    print(f"可用的执行提供程序: {ort.get_available_providers()}")
    
    # PyTorch版本
    print(f"\nPyTorch版本: {torch.__version__}")

def init_pose_models():
    """初始化两个姿势检测模型"""
    rf = roboflow.Roboflow()
    workspace = rf.workspace("bigblacklee")
    
    # 初始化全身模型
    full_body_model = workspace.project("draft-estimation").version(2).model
    full_body_model.confidence = 20  # 设置较低的置信度阈值
    
    # 初始化复杂姿势模型
    complex_pose_model = workspace.project("posedraft-estimation-1").version(8).model
    complex_pose_model.confidence = 20
    
    return full_body_model, complex_pose_model

class SketchVideoExtractor:
    def __init__(self):
        """初始化视频提取器"""
        self.threshold = 30  # 检测变化的阈值
        self.min_area = 100  # 最小变化区域
        self.frame_buffer = []  # 帧缓冲
        self.keyframes = []  # 关键帧列表
        self.timestamps = []  # 时间戳列表
        
        # 初始化两个模型
        print("\n初始化姿势检测模型...")
        self.full_body_model, self.complex_pose_model = init_pose_models()
        self.confidence_threshold = 0.2
        print("模型初始化完成!")
        
    def set_parameters(self, threshold=30, min_area=100):
        """设置检测参数"""
        self.threshold = threshold
        self.min_area = min_area
        
    def detect_change(self, frame1, frame2):
        """检测两帧之间的变化"""
        try:
            # 转换为灰度图
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = frame1, frame2
                
            # 计算差异
            diff = cv2.absdiff(gray1, gray2)
            
            # 应用阈值
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # 查找变化区域
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 计算变化区域的总面积
            total_area = sum(cv2.contourArea(c) for c in contours)
            
            return total_area > self.min_area, total_area
            
        except Exception as e:
            print(f"检测变化时出错: {str(e)}")
            return False, 0
            
    def detect_pose(self, image_path):
        """使用两个模型进行级联检测"""
        try:
            # 首先使用全身模型
            full_body_pred = self.full_body_model.predict(str(image_path))
            full_body_results = full_body_pred.json()
            
            # 检查全身模型是否检测到姿势
            if (full_body_results.get('predictions', []) and 
                full_body_results['predictions'][0].get('predictions', [])):
                print(f"\n全身模型检测到姿势: {image_path}")
                return {
                    'model_used': 'full_body',
                    'results': full_body_results
                }
            
            print(f"\n全身模型未检测到姿势，尝试复杂姿势模型: {image_path}")
            # 使用复杂姿势模型
            complex_pred = self.complex_pose_model.predict(str(image_path))
            complex_results = complex_pred.json()
            
            # 检查复杂姿势模型是否检测到
            if (complex_results.get('predictions', []) and 
                complex_results['predictions'][0].get('predictions', [])):
                print(f"复杂姿势模型检测到姿势: {image_path}")
                return {
                    'model_used': 'complex_pose',
                    'results': complex_results
                }
            
            print(f"两个模型都未检测到姿势: {image_path}")
            return None
            
        except Exception as e:
            print(f"姿势检测时出错 {image_path}: {str(e)}")
            return None
            
    def visualize_pose(self, image_path, predictions, output_path):
        """可视化姿势检测结果"""
        try:
            # 首先打印接收到的预测结果
            print("\n接收到的预测结果：")
            print(predictions)
            
            image = cv2.imread(str(image_path))
            
            # 确保predictions是列表
            if isinstance(predictions, dict):
                predictions = [predictions]
            
            # 处理预测结果
            for prediction in predictions:
                try:
                    # 获取边界框信息（添加错误处理）
                    if isinstance(prediction, dict):
                        x = float(prediction.get('x', 0))
                        y = float(prediction.get('y', 0))
                        width = float(prediction.get('width', 0))
                        height = float(prediction.get('height', 0))
                        confidence = float(prediction.get('confidence', 0))
                        
                        # 计算边界框坐标
                        x1 = int(x - width/2)
                        y1 = int(y - height/2)
                        x2 = int(x + width/2)
                        y2 = int(y + height/2)
                        
                        # 绘制边界框
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, f"{confidence:.2f}",
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 绘制关键点（如果有）
                        if 'keypoints' in prediction:
                            for keypoint in prediction['keypoints']:
                                if isinstance(keypoint, dict):
                                    kp_x = int(keypoint.get('x', 0))
                                    kp_y = int(keypoint.get('y', 0))
                                    cv2.circle(image, (kp_x, kp_y), 3, (255, 0, 0), -1)
                
                except Exception as e:
                    print(f"处理单个预测时出错: {str(e)}")
                    continue
            
            cv2.imwrite(str(output_path), image)
            
        except Exception as e:
            print(f"可视化结果时出错: {str(e)}")
    
    def extract_frames(self, video_path, output_dir):
        """从视频中提取关键帧并进行姿势检测"""
        try:
            # 创建输出目录结构
            output_path = Path(output_dir)
            full_body_dir = output_path / "full_body"  # 全身模型结果目录
            complex_pose_dir = output_path / "complex_pose"  # 复杂姿势模型结果目录
            
            # 创建必要的目录
            for dir_path in [full_body_dir, complex_pose_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 打开视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
                
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"\n视频信息:")
            print(f"- FPS: {fps}")
            print(f"- 总帧数: {total_frames}")
            
            # 读取第一帧
            ret, prev_frame = cap.read()
            if not ret:
                raise ValueError("无法读取视频帧")
                
            frame_count = 1
            last_keyframe_time = 0
            
            # 保存第一帧
            first_frame_path = full_body_dir / f"frame_0001.png"
            cv2.imwrite(str(first_frame_path), prev_frame)
            self.keyframes.append(str(first_frame_path))
            self.timestamps.append(0.0)
            frame_numbers = [1]  # 添加帧数列表，从第一帧开始
            
            print("\n开始处理视频...")
            
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                current_time = frame_count / fps
                
                # 检测变化
                has_change, change_area = self.detect_change(prev_frame, current_frame)
                
                if has_change and (current_time - last_keyframe_time) >= 0.5:  # 最小间隔0.5秒
                    # 保存关键帧
                    frame_path = full_body_dir / f"frame_{frame_count:04d}.png"
                    cv2.imwrite(str(frame_path), current_frame)
                    
                    self.keyframes.append(str(frame_path))
                    self.timestamps.append(current_time)
                    frame_numbers.append(frame_count)  # 记录帧数
                    
                    last_keyframe_time = current_time
                    
                    print(f"检测到变化 - 帧 {frame_count}, 时间: {current_time:.2f}秒, 变化区域: {change_area:.2f}")
                    
                prev_frame = current_frame.copy()
                
                # 显示处理进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"处理进度: {progress:.1f}%", end='\r')
                    
            cap.release()
            
            # 进行姿势检测
            print("\n\n开始进行姿势检测...")
            pose_results = {
                'full_body': {},
                'complex_pose': {}
            }
            detection_summary = {
                'full_body_detections': 0,
                'complex_pose_detections': 0,
                'no_detections': 0
            }
            
            # 修改关键帧信息的收集
            keyframe_info = []
            for i, frame_path in enumerate(self.keyframes):
                frame_number = int(Path(frame_path).stem.split('_')[1])
                
                # 检测姿势
                predictions = self.detect_pose(frame_path)
                
                # 构建帧信息
                frame_info = {
                    "frame_number": frame_number,
                    "timestamp": self.timestamps[i],
                    "file_path": str(frame_path)
                }
                
                # 添加预测结果
                if predictions and 'results' in predictions:
                    frame_info["predictions"] = {
                        "model_used": predictions["model_used"],
                        "results": predictions["results"]  # 保持完整的预测结果
                    }
                    
                    # 更新统计信息
                    if predictions["model_used"] == "full_body":
                        detection_summary["full_body_detections"] += 1
                    else:
                        detection_summary["complex_pose_detections"] += 1
                else:
                    detection_summary["no_detections"] += 1
                
                keyframe_info.append(frame_info)
            
            # 构建完整的检测信息
            detection_info = {
                "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_frames": len(self.keyframes),
                "detection_summary": detection_summary,
                "keyframe_info": keyframe_info
            }
            
            # 保存检测信息
            detection_info_path = Path(output_dir) / "detection_info.json"
            with open(detection_info_path, 'w', encoding='utf-8') as f:
                json.dump(detection_info, f, ensure_ascii=False, indent=2)
            
            print(f"\n\n处理完成!")
            print(f"- 提取的关键帧数量: {len(self.keyframes)}")
            print(f"- 全身姿势检测数量: {detection_summary['full_body_detections']}")
            print(f"- 复杂姿势检测数量: {detection_summary['complex_pose_detections']}")
            print(f"- 未检测到姿势数量: {detection_summary['no_detections']}")
            print(f"- 输出目录: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"处理时出错: {str(e)}")
            return False

def main():
    """主函数"""
    # 检查环境
    check_versions()
    
    # 创建提取器实例
    extractor = SketchVideoExtractor()
    
    # 设置参数
    extractor.set_parameters(threshold=30, min_area=100)
    
    # 设置输入输出路径
    current_dir = Path(__file__).parent
    video_dir = current_dir / "input" / "video"
    output_dir = current_dir / "output" / "seperate_frame"
    
    # 检查视频目录是否存在
    if not video_dir.exists():
        print(f"错误：视频目录不存在: {video_dir}")
        return
        
    # 查找视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
    
    if not video_files:
        print(f"错误：在 {video_dir} 中未找到视频文件")
        print("支持的格式:", video_extensions)
        return
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个视频文件
    for video_path in video_files:
        print(f"\n处理视频: {video_path.name}")
        # 为每个视频创建单独的输出目录
        video_output_dir = output_dir / video_path.stem
        video_output_dir.mkdir(exist_ok=True)
        
        # 提取关键帧并进行姿势检测
        extractor.extract_frames(str(video_path), str(video_output_dir))

if __name__ == "__main__":
    main()