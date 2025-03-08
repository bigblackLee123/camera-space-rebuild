
import bpy
import math

# 设置相机位置
bpy.data.objects['Camera'].location = (0.01864880430109518, 0.07503949922489236, -0.9047703905846901)

# 设置相机旋转（欧拉角，弧度制）
bpy.data.objects['Camera'].rotation_euler = (1.7655322070467834, -0.03186302422896851, 0.007924907778587337)

# 设置相机内参
bpy.data.cameras['Camera'].lens = 995.5555555555555  # 焦距
