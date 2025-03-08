# 相机估计与绘画辅助工具

这是一个用于视频处理、姿态检测、深度图生成和相机参数估计的综合工具集，可以帮助艺术家和动画师更轻松地进行3D建模和动画制作
![show_1](https://github.com/user-attachments/assets/1b34ffd5-108a-48bb-9952-f7260f0fac5e)



## 功能特点

- **视频处理**：从视频中提取关键帧
- **姿态检测**：使用先进的AI模型检测人体姿态
- **深度图生成**：基于Depth-Anything-V2生成高质量深度图
- **相机参数估计**：计算并导出相机内外参数到Blender
- **骨骼映射**：将检测到的姿态映射到MMD模型骨骼
- **音频生成**：集成音频处理和生成功能
- **用户友好界面**：基于PySide6的现代化图形界面

## 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐用于深度图生成和姿态检测）
- Blender 3.0+（用于3D模型处理和相机设置）

## 安装指南

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/camera_estimate.git
cd camera_estimate
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载模型权重：
   - 创建`checkpoints`目录
   - 下载Depth-Anything-V2模型权重并放入`checkpoints`目录
   - 模型权重下载链接：[Depth-Anything-V2](https://github.com/depth-anything/depth-anything-v2)

4. 安装Blender（如果尚未安装）

## 使用方法

### 视频处理与姿态检测

1. 运行主界面：
```bash
python ui.0124.py
```

2. 加载视频文件并使用"分离视频"功能提取关键帧
3. 使用"生成深度图"功能为提取的帧生成深度信息

![show_2](https://github.com/user-attachments/assets/9e1d4dc4-1896-49ae-8df2-fdae7e2f2c6b)

4. 使用"标注草图"功能手动调整或标注关键点
   
![show_3](https://github.com/user-attachments/assets/137e6905-a611-419e-b873-346c933131e6)

### 相机参数估计与Blender导出

1. 完成姿态检测和深度图生成后，打开blender，加载mmd模型，在blender中运行skeleton_detect文件，检测mmd模型骨骼位置

2. 运行space_rebuild.py,标注后草图骨骼关键点信息，计算每帧骨骼与相机相对位置

3.运行apply_to_blender.py，读取计算的每帧骨骼与相机相对位置，进行相机截图，返回截图到程序中
![show_5](https://github.com/user-attachments/assets/8d0bbc1b-9427-4c1d-9ac1-91a161a45dd8)


### 音频处理

1. 在主界面中点击"音频生成器"按钮
2. 使用音频生成界面处理和生成音频
   
![show_4](https://github.com/user-attachments/assets/0ff8fb5c-5a12-4387-951d-9e7793064aef)

## 项目结构

- `ui.py` - 主用户界面
- `video_sep.py` - 视频处理和帧提取
- `depth.py` - 深度图生成
- `depth_map.py` - 深度图处理和可视化
- `skeleton_detect.py` - 骨骼检测和MMD模型导出
- `apply_to_blender.py` - 相机参数计算和Blender导出
- `audio_generate.py` - 音频处理和生成
- `logger.py` - 日志记录工具

## 注意事项

- 确保在使用前已正确安装所有依赖
- 深度图生成需要较大的GPU内存
- 使用Blender功能时，确保Blender已正确安装并可从Python访问
