
import bpy

# ���������
bpy.ops.object.camera_add()
camera = bpy.context.object
camera.name = "EstimatedCamera"

# �����ڲ�
camera.data.lens = 28.0
camera.data.sensor_width = 36
camera.data.sensor_height = 20.25

# �������
camera.location = (0, 0, -5)
camera.rotation_euler = (1, 0, 0)

print("������������")
