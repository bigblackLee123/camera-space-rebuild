import bpy
import math
import numpy as np

def create_camera():
    # 确保在正确的上下文中操作
    if bpy.context.active_object:
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    
    # 创建新相机
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    
    # 将相机添加到场景集合
    scene = bpy.context.scene
    scene.collection.objects.link(camera_object)
    
    # 设置相机位置
    camera_object.location = ([-6.84452298e-17], [-1.33889601e-18], [-9.3530618e-17])
    
    # 设置相机旋转（欧拉角，角度制）
    camera_object.rotation_euler = (
        math.radians(-8.195369439050756e-16),
        math.radians(0.0),
        math.radians(2.0316358792240838e-14)
    )
    
    # 设置相机参数
    camera_object.data.lens = 28.0  # 焦距（毫米）
    camera_object.data.sensor_width = 36  # 传感器宽度（毫米）
    
    # 选择并激活相机
    camera_object.select_set(True)
    bpy.context.view_layer.objects.active = camera_object
    
    # 将新创建的相机设为场景的活动相机
    scene.camera = camera_object
    
    return camera_object

def main():
    try:
        # 创建相机
        camera = create_camera()
        print(f"相机已创建：{camera.name}")
        print(f"位置：{camera.location}")
        print(f"旋转：{[math.degrees(a) for a in camera.rotation_euler]}")
        print(f"焦距：{camera.data.lens}mm")
        print(f"传感器宽度：{camera.data.sensor_width}mm")
    except Exception as e:
        print(f"创建相机时出错：{str(e)}")

if __name__ == "__main__":
    main()
