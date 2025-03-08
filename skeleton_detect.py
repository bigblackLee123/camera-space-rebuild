import bpy
import json
import os
import sys
from mathutils import Vector, Quaternion
from typing import Dict, List, Optional, Any
import traceback

class MMDModelExporter:
    def __init__(self):
        # 首先设置logger
        self.logger = self.setup_logger()
        # 然后获取armature
        self.armature = self.get_mmd_armature()
        self.bones_data = {}
        
        # 获取当前工程文件所在目录
        blend_file_path = bpy.data.filepath
        project_dir = os.path.dirname(blend_file_path)
        # 设置输出路径为当前目录下的output文件夹
        self.output_path = os.path.join(project_dir, 'output')
        os.makedirs(self.output_path, exist_ok=True)
        self.logger.info(f"输出路径设置为: {self.output_path}")
    
    def setup_logger(self):
        """设置日志"""
        import logging
        
        # 创建logger
        logger = logging.getLogger('MMDExporter')
        logger.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        log_path = os.path.join(os.path.dirname(bpy.data.filepath), "mmd_export.log")
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def get_mmd_armature(self) -> Optional[bpy.types.Object]:
        """获取场景中的MMD骨骼"""
        try:
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE' and hasattr(obj, 'mmd_root'):
                    self.logger.info(f"找到MMD模型: {obj.name}")
                    return obj
            
            self.logger.warning("未找到MMD模型")
            return None
            
        except Exception as e:
            self.logger.error(f"获取MMD骨骼时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def get_bone_properties(self, bone):
        """获取骨骼位置和长度信息"""
        try:
            head_pos = tuple(bone.head)
            tail_pos = tuple(bone.tail)
            # 计算骨骼长度（头尾点之间的距离）
            length = ((tail_pos[0] - head_pos[0])**2 + 
                     (tail_pos[1] - head_pos[1])**2 + 
                     (tail_pos[2] - head_pos[2])**2) ** 0.5
            
            return {
                'name': bone.name,
                'head_position': head_pos,  # 骨骼起始点位置
                'tail_position': tail_pos,  # 骨骼结束点位置
                'length': length            # 骨骼长度
            }
            
        except Exception as e:
            self.logger.error(f"获取骨骼属性时出错: {str(e)}")
            return None
    
    def get_bone_constraints(self, bone: bpy.types.PoseBone) -> List[Dict[str, Any]]:
        """获取骨骼约束"""
        try:
            constraints = []
            for constraint in bone.constraints:
                constraint_data = {
                    'name': constraint.name,
                    'type': constraint.type,
                    'influence': constraint.influence,
                    'mute': constraint.mute
                }
                
                # 根据约束类型获取特定属性
                if constraint.type == 'IK':
                    constraint_data.update({
                        'target': constraint.target.name if constraint.target else None,
                        'pole_target': constraint.pole_target.name if constraint.pole_target else None,
                        'iterations': constraint.iterations,
                        'chain_count': constraint.chain_count
                    })
                
                constraints.append(constraint_data)
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"获取骨骼约束时出错: {str(e)}")
            return []
    
    def export_mmd_bones(self) -> Dict[str, Any]:
        """导出MMD骨骼数据"""
        try:
            if not self.armature:
                self.logger.error("未找到MMD模型，无法导出骨骼数据")
                return {}
            
            bones_data = {}
            
            # 遍历所有骨骼
            for bone in self.armature.pose.bones:
                try:
                    # 获取全局变换
                    world_matrix = self.armature.matrix_world @ bone.matrix
                    
                    # 获取位置、旋转和缩放
                    location = world_matrix.to_translation()
                    rotation = world_matrix.to_quaternion()
                    scale = world_matrix.to_scale()
                    
                    # 获取骨骼数据
                    bone_data = {
                        'name_jp': bone.name,
                        'name_en': bone.get('name_en', ''),
                        'transform': {
                            'location': [location.x, location.y, location.z],
                            'rotation': [rotation.w, rotation.x, rotation.y, rotation.z],
                            'scale': [scale.x, scale.y, scale.z],
                            'matrix_local': [list(row) for row in bone.bone.matrix_local],
                            'head': list(bone.bone.head),
                            'tail': list(bone.bone.tail)
                        },
                        'hierarchy': {
                            'parent': bone.parent.name if bone.parent else None,
                            'children': [child.name for child in bone.children]
                        },
                        'properties': self.get_bone_properties(bone),
                        'constraints': self.get_bone_constraints(bone)
                    }
                    
                    bones_data[bone.name] = bone_data
                    
                except Exception as e:
                    self.logger.error(f"处理骨骼 {bone.name} 时出错: {str(e)}")
                    continue
            
            self.bones_data = bones_data
            return bones_data
            
        except Exception as e:
            self.logger.error(f"导出骨骼数据时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def is_human_bone(self, bone_name):
        """判断是否为需要的骨骼"""
        target_bones = {
            'ひざ.R',      # 右膝
            'ひざ.L',      # 左膝
            '足.R',        # 右脚
            '足.L',        # 左脚
            '下半身',      # 下半身
            '上半身',      # 上半身
            '胸上L',       # 左胸下
            '胸上R',       # 右胸下
            '肩R',         # 右肩
            '肩L',         # 左肩
            '腕R',         # 右臂
            '腕L',         # 左臂
            'ひじ.R',      # 右肘
            'ひじ.L',      # 左肘
            '上半身2',     # 上半身2
            '首',          # 脖子
            '肩.R',        # 右肩（补充）
            '肩.L',        # 左肩（补充）
            '腕.R',        # 右臂（补充）
            '腕.L',        # 左臂（补充）
            '胸上.L',      # 左胸上（补充）
            '胸上.R',      # 右胸上（补充）
            '足D.R',      # 右胯
            '足D.L',      # 左胯
        }
        return bone_name in target_bones

    def save_mmd_data(self):
        """保存MMD模型数据"""
        try:
            if self.armature and self.armature.type == 'ARMATURE':
                # 获取所有骨骼数据
                bones_data = {}
                for bone in self.armature.data.bones:
                    # 只处理人体骨骼
                    if self.is_human_bone(bone.name):
                        bone_props = self.get_bone_properties(bone)
                        if bone_props:
                            bones_data[bone.name] = bone_props

                # 准备输出文件路径
                output_file = os.path.join(self.output_path, 'mmd_bones_data.json')
                
                # 保存为JSON文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(bones_data, f, ensure_ascii=False, indent=4)
                
                self.logger.info(f"MMD骨骼数据已保存到: {output_file}")
                return True
                
            return False
                
        except Exception as e:
            self.logger.error(f"保存MMD数据时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

def main():
    """主函数"""
    try:
        # 创建导出器实例
        exporter = MMDModelExporter()
        
        # 导出数据
        success = exporter.save_mmd_data()
        
        if success:
            print("MMD模型数据导出成功！")
        else:
            print("MMD模型数据导出失败！")
            
    except Exception as e:
        print(f"导出过程出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()