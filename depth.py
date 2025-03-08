import cv2
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
current_dir = Path(__file__).parent

# 添加Depth-Anything目录到Python路径
depth_anything_path = current_dir / "Depth-Anything-V2"
if depth_anything_path.exists():
    sys.path.append(str(depth_anything_path))
else:
    raise FileNotFoundError(f"找不到Depth-Anything目录: {depth_anything_path}")

# 现在可以导入depth_anything_v2
from depth_anything_v2.dpt import DepthAnythingV2

class DepthGenerator:
    def __init__(self):
        """初始化深度图生成器"""
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 模型配置
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # 加载模型
        self.encoder = 'vitl'  # 默认使用大型模型
        self.model = self._load_model()
        
    def _load_model(self):
        """加载Depth Anything V2模型"""
        try:
            print(f"正在加载 {self.encoder} 模型...")
            model = DepthAnythingV2(**self.model_configs[self.encoder])
            model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
            model = model.to(self.device).eval()
            print("模型加载完成")
            return model
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise

    def generate_depth(self, image_path, output_path=None):
        """生成深度图"""
        try:
            # 读取图像
            if isinstance(image_path, str):
                image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"找不到图像文件: {image_path}")
                
            raw_image = cv2.imread(str(image_path))
            if raw_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 生成深度图
            with torch.no_grad():
                depth = self.model.infer_image(raw_image)
            
            # 归一化到0-255范围
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            # 保存深度图
            if output_path:
                cv2.imwrite(str(output_path), depth)
                print(f"深度图已保存至: {output_path}")
            
            return depth
            
        except Exception as e:
            print(f"深度图生成出错: {str(e)}")
            return None

    def process_video_frames(self, seperate_frame_dir):
        """处理video_sep输出的帧"""
        try:
            seperate_frame_dir = Path(seperate_frame_dir)
            if not seperate_frame_dir.exists():
                raise FileNotFoundError(f"找不到目录: {seperate_frame_dir}")

            # 读取detection_info.json
            info_path = seperate_frame_dir / "detection_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"找不到detection_info.json: {info_path}")

            with open(info_path, 'r', encoding='utf-8') as f:
                detection_info = json.load(f)

            # 处理full_body和complex_pose目录
            for model_type in ['full_body', 'complex_pose']:
                model_dir = seperate_frame_dir / model_type
                if not model_dir.exists():
                    continue

                # 遍历所有帧目录
                frame_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                total_frames = len(frame_dirs)
                
                print(f"\n处理 {model_type} 目录中的帧...")
                for i, frame_dir in enumerate(frame_dirs, 1):
                    print(f"处理帧 [{i}/{total_frames}]: {frame_dir.name}")
                    
                    # 读取原始帧图像
                    frame_path = frame_dir / f"{frame_dir.name}.png"
                    if not frame_path.exists():
                        print(f"警告：找不到帧图像: {frame_path}")
                        continue
                    
                    # 生成深度图
                    depth_path = frame_dir / f"{frame_dir.name}_depth.png"
                    depth = self.generate_depth(frame_path, depth_path)
                    
                    if depth is not None:
                        # 更新JSON结果文件
                        json_path = frame_dir / f"{frame_dir.name}_results.json"
                        if json_path.exists():
                            with open(json_path, 'r', encoding='utf-8') as f:
                                results = json.load(f)
                            
                            # 添加深度图信息
                            results['depth_map'] = {
                                'path': str(depth_path),
                                'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=2)

            print("\n所有帧处理完成!")
            
        except Exception as e:
            print(f"处理目录时出错: {str(e)}")

def main():
    """主函数"""
    # 创建深度图生成器实例
    generator = DepthGenerator()
    
    # 设置输入目录
    current_dir = Path(__file__).parent
    seperate_frame_dir = current_dir / "output" / "seperate_frame"
    
    # 处理所有视频目录
    for video_dir in seperate_frame_dir.iterdir():
        if video_dir.is_dir():
            print(f"\n处理视频目录: {video_dir.name}")
            generator.process_video_frames(video_dir)

if __name__ == "__main__":
    main()