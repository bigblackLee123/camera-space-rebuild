from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sys
import cv2
from pathlib import Path
from logger import Logger
import json
import traceback

# 创建全局logger实例
logger = Logger()

class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("初始化视频编辑器...")
        self.setWindowTitle("视频编辑器")
        self.setFixedSize(1920, 1080)

        # 创建主窗口部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 设置顶部栏
        self.setup_top_bar()
        
        # 状态栏 (1920x32)
        self.status_bar = QWidget()
        self.status_bar.setFixedSize(1920, 32)
        self.status_bar.setStyleSheet("""
            QWidget {
                background-color: #CCCCCC;
                border: 1px solid #999999;
                border-radius: 10px;
            }
        """)
        
        # 主内容区域
        self.content = QWidget()
        self.content_layout = QHBoxLayout(self.content)
        self.content_layout.setSpacing(24)
        self.content_layout.setContentsMargins(28, 15, 28, 15)
        
        # 左侧功能区 (341x966)
        self.left_panel = QTabWidget()
        self.left_panel.setFixedWidth(341)
        self.left_panel.setStyleSheet("""
            QTabWidget {
                background-color: transparent;
                border: none;
            }
            QTabWidget::pane {
                background-color: #CCCCCC;
                border-radius: 10px;
                border: 1px solid #999999;
            }
            QTabBar::tab {
                background-color: #BBBBBB;
                padding: 5px 15px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background-color: #CCCCCC;
            }
            QTabBar {
                background-color: transparent;
            }
        """)
        
        # 添加两个标签页
        self.action_tab = QWidget()
        self.preview_tab = QWidget()
        self.left_panel.addTab(self.action_tab, "动作检测")
        self.left_panel.addTab(self.preview_tab, "标记预览")
        
        # 标签页内容区域样式
        self.action_tab.setStyleSheet("""
            QWidget {
                background-color: #CCCCCC;
            }
        """)
        self.preview_tab.setStyleSheet("""
            QWidget {
                background-color: #CCCCCC;
            }
        """)
        
        # 右侧内容区域
        self.right_content = QWidget()
        self.right_layout = QVBoxLayout(self.right_content)
        self.right_layout.setSpacing(24)
        
        # 上方预览区域
        self.preview_area = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_area)
        self.preview_layout.setSpacing(24)
        
        # 草图预览 (721x480)
        self.sketch_preview = QLabel("草图预览")
        self.sketch_preview.setFixedSize(721, 480)
        self.sketch_preview.setStyleSheet("""
            QLabel {
                background-color: #CCCCCC;
                border: 1px solid #999999;
                border-radius: 10px;
            }
        """)
        
        # Blender预览 (721x480)
        self.blender_preview = QLabel("Blender预览")
        self.blender_preview.setFixedSize(721, 480)
        self.blender_preview.setStyleSheet("""
            QLabel {
                background-color: #2B2B2B;
                border: 1px solid #999999;
                border-radius: 10px;
                color: white;
            }
        """)
        
        # 下方视频区域
        # 原视频预览 (1465x212)
        self.video_preview = QLabel("原视频预览")
        self.video_preview.setFixedSize(1465, 212)
        self.video_preview.setStyleSheet("""
            QLabel {
                background-color: #CCCCCC;
                border: 1px solid #999999;
                border-radius: 10px;
            }
        """)
        
        # Blender视频预览 (1465x212)
        self.blender_video = QLabel("Blender视频预览")
        self.blender_video.setFixedSize(1465, 212)
        self.blender_video.setStyleSheet("""
            QLabel {
                background-color: #CCCCCC;
                border: 1px solid #999999;
                border-radius: 10px;
            }
        """)
        
        # 设置动作检测标签页
        self.setup_action_tab()
        
        # 设置预览标签页
        self.setup_preview_tab()
        
        # 初始化Blender预览相关变量
        self.blender_path = Path("F:/ai_program_2/camera_estimate/workfile/paint_helper/output/seperate_frame/test_0109/blender_scene")
        
        # 设置Blender预览窗口
        self.blender_preview.setAlignment(Qt.AlignCenter)
        self.blender_preview.setStyleSheet("""
            QLabel {
                background-color: #2B2B2B;
                border: 1px solid #999999;
                border-radius: 10px;
                color: white;
            }
        """)
        
        # 创建Blender时间线布局
        self.blender_timeline_widget = QWidget()
        self.blender_timeline_widget.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border: 1px solid #999999;
                border-radius: 10px;
            }
        """)
        
        self.blender_timeline_layout = QHBoxLayout(self.blender_timeline_widget)
        self.blender_timeline_layout.setContentsMargins(10, 5, 10, 5)
        self.blender_timeline_layout.setSpacing(5)
        
        # 创建滚动区域
        self.blender_scroll = QScrollArea()
        self.blender_scroll.setWidget(self.blender_timeline_widget)
        self.blender_scroll.setWidgetResizable(True)
        self.blender_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.blender_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.blender_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:horizontal {
                height: 12px;
                background: #2B2B2B;
            }
            QScrollBar::handle:horizontal {
                background: #666666;
                min-width: 20px;
                border-radius: 6px;
            }
        """)
        
        # 替换原来的blender_video
        self.blender_video.setLayout(QVBoxLayout())
        self.blender_video.layout().setContentsMargins(0, 0, 0, 0)
        self.blender_video.layout().addWidget(self.blender_scroll)
        
        # 初始化时更新时间线
        self.update_blender_timeline()
        
        # 组装界面
        self.assemble_ui()
        
        # 视频相关属性
        self.video_path = None
        self.video_cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 0
        
        # 创建视频控制组件
        self.setup_video_controls()
    
    def setup_top_bar(self):
        """设置顶部栏"""
        # 创建顶部栏容器
        self.top_bar = QWidget()
        self.top_bar.setFixedHeight(40)  # 设置固定高度
        self.top_bar.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border-bottom: 1px solid #999999;
            }
            QPushButton {
                background-color: #3B3B3B;
                border: 1px solid #555555;
                border-radius: 4px;
                color: white;
                padding: 5px 15px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #4B4B4B;
            }
        """)

        # 创建水平布局
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(10, 0, 10, 0)

        # 创建两个按钮
        #self.load_btn = QPushButton("加载视频")
        self.process_btn = QPushButton("草图标注")
        self.audio_generate_btn = QPushButton("音频生成")
    
        # 连接按钮信号
        #self.load_btn.clicked.connect(self.load_video_file)
        self.process_btn.clicked.connect(self.run_sketch_annotation)
        self.audio_generate_btn.clicked.connect(self.open_audio_generator)
    
        # 添加按钮到布局
        #top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.process_btn)
        top_layout.addWidget(self.audio_generate_btn)
        top_layout.addStretch()  # 添加弹性空间
    
        # 将顶部栏添加到主布局
        self.layout.insertWidget(0, self.top_bar)  # 插入到布局最上方
        
    def setup_action_tab(self):
        """设置动作检测标签页"""
        layout = QVBoxLayout(self.action_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 添加加载视频按钮
        load_btn = QPushButton("加载视频")
        load_btn.setFixedSize(120, 30)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        load_btn.clicked.connect(self.load_video_file)
        
        # 添加动作检测按钮
        detect_btn = QPushButton("运行视频检测")
        detect_btn.setFixedSize(120, 30)
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        detect_btn.clicked.connect(self.run_video_sep)
        
        # 添加深度图生成按钮
        depth_btn = QPushButton("生成深度图")
        depth_btn.setFixedSize(120, 30)
        depth_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        depth_btn.clicked.connect(self.generate_depth_maps)
        
        # 按钮容器（用于水平居中）
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setSpacing(10)  # 按钮之间的间距
        
        # 添加按钮到容器
        for btn in [load_btn, detect_btn, depth_btn]:
            btn_wrapper = QWidget()
            wrapper_layout = QHBoxLayout(btn_wrapper)
            wrapper_layout.addStretch()
            wrapper_layout.addWidget(btn)
            wrapper_layout.addStretch()
            btn_layout.addWidget(btn_wrapper)
        
        # 添加到主布局
        layout.addWidget(btn_container)
        layout.addStretch()  # 添加弹性空间
        
    def setup_preview_tab(self):
        """设置预览标签页"""
        layout = QVBoxLayout(self.preview_tab)
        
        # 添加预览列表
        self.preview_list = QListWidget()
        
        layout.addWidget(self.preview_list)
    
    def run_sketch_annotation(self):
        """运行草图标注"""
        try:
            if hasattr(self, 'annotator'):
                # 如果已经存在窗口，先关闭它
                self.annotator.close_window()
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            from depth_map import SketchAnnotator, COCOKeypointProcessor
            self.annotator = SketchAnnotator()
            self.keypoint_processor = COCOKeypointProcessor()
            
            if self.video_cap is None:
                QMessageBox.warning(self, "警告", "请先加载视频！")
                return
            
            # 初始化标注器和关键点处理器
            self.update_annotation_frame(self.current_frame)
            
            window_name = "草图标注工具"
            cv2.namedWindow(window_name)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            # 设置鼠标回调
            callback_params = {
                'image': self.annotator.image,
                'window_name': window_name
            }
            cv2.setMouseCallback(window_name, 
                                self.keypoint_processor.mouse_callback, 
                                callback_params)

            
            # 显示初始图像
            if self.keypoint_processor.keypoints:
                annotated_frame = self.keypoint_processor.draw_on_sketch(self.annotator.image)
                cv2.imshow(window_name, annotated_frame)
            
            # 添加标志来控制循环
            self.annotation_running = True
            
            # 主循环
            while self.annotation_running:
                # 检查Qt窗口是否关闭
                if not self.isVisible():
                    self.annotation_running = False
                    break
                
                # 检查OpenCV窗口是否关闭
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self.annotation_running = False
                        break
                except cv2.error:
                    self.annotation_running = False
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # q或ESC退出
                    self.annotation_running = False
                    break
                elif key == ord('b'):  # 按b导出关键点
                    try:
                        # 确保标注数据已更新
                        self.annotator.points = self.keypoint_processor.keypoints
                        self.annotator.lines = self.keypoint_processor.skeleton
                        
                        video_name = Path(self.video_path).stem
                        base_dir = "output/seperate_frame"
                        self.annotator.export_keypoints(base_dir, video_name, self.current_frame,model_type="complex_pose")
                        self.annotator.export_keypoints(base_dir, video_name, self.current_frame,model_type="full_body")
                        
                    except Exception as e:
                        logger.error(f"导出关键点时出错：{str(e)}")
                        QMessageBox.critical(self, "错误", f"导出关键点时出错：{str(e)}")
            
        except Exception as e:
            logger.error(f"启动草图标注时出错：{str(e)}")
            QMessageBox.critical(self, "错误", f"启动草图标注时出错：{str(e)}")
        finally:
            # 确保资源被正确释放
            self.annotation_running = False
            if hasattr(self, 'annotator'):
                self.annotator.is_running = False
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def update_annotation_frame(self, frame_number):
        """更新草图标注窗口的帧"""
        try:
            if not hasattr(self, 'annotator') or self.video_cap is None:
                return
            
            # 读取视频帧
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video_cap.read()
            if not ret:
                logger.error(f"无法读取帧 {frame_number}")
                return
            
            # 重置原始图像
            if hasattr(self.keypoint_processor, 'original_image'):
                delattr(self.keypoint_processor, 'original_image')
            
            # 更新图像
            self.annotator.load_frame(frame)
            self.keypoint_processor.current_image = self.annotator.image

            # 只在首次加载或切换到新帧时加载关键点数据
            if (not self.keypoint_processor.keypoints or 
                self.current_frame != frame_number):
                
                self.keypoint_processor.keypoints = None  # 清除现有关键点
                self.keypoint_processor.skeleton = None   # 清除现有骨架
                
                # 加载新帧的关键点数据
                if self.detection_info:
                    current_frame_info = next(
                        (frame for frame in self.detection_info['keyframe_info'] 
                         if frame['frame_number'] == frame_number), 
                        None
                    )
                    
                    if current_frame_info and 'predictions' in current_frame_info:
                        predictions = current_frame_info['predictions']
                        model_type = predictions.get('model_used')
                        results = predictions.get('results')
                        
                        if model_type and results:
                            logger.info(f"处理帧 {frame_number} 的 {model_type} 数据")
                            try:
                                if model_type == "complex_pose":
                                    self.keypoint_processor.load_complex_pose_data(results)
                                elif model_type == "full_body":
                                    self.keypoint_processor.load_full_body_data(results)
                                else:
                                    logger.warning(f"未知的模型类型: {model_type}")
                            except Exception as e:
                                logger.error(f"加载关键点数据时出错: {str(e)}")
                                traceback.print_exc()
            
            # 更新显示
            if self.keypoint_processor.keypoints:
                annotated_frame = self.keypoint_processor.draw_on_sketch(frame)
                self.annotator.load_frame(annotated_frame)
                
                try:
                    if cv2.getWindowProperty(self.annotator.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.imshow(self.annotator.window_name, annotated_frame)
                        cv2.waitKey(1)
                except cv2.error:
                    pass
                
        except Exception as e:
            logger.error(f"更新帧时出错: {str(e)}")
            traceback.print_exc()
    

    def assemble_ui(self):
        """组装界面"""
        # 添加预览区域组件
        self.preview_layout.addWidget(self.sketch_preview)
        self.preview_layout.addWidget(self.blender_preview)
        
        # 添加右侧内容
        self.right_layout.addWidget(self.preview_area)
        self.right_layout.addWidget(self.video_preview)
        self.right_layout.addWidget(self.blender_video)
        
        # 添加左右面板
        self.content_layout.addWidget(self.left_panel)
        self.content_layout.addWidget(self.right_content)
        
        # 添加到主布局
        self.layout.addWidget(self.status_bar)
        self.layout.addWidget(self.content)
        
    def run_video_sep(self):
        """运行视频检测"""
        try:
            from video_sep import SketchVideoExtractor
            
            # 设置默认目录为项目的input文件夹
            current_dir = Path(__file__).parent
            input_dir = current_dir / "input" / "video"
            
            # 确保input目录存在
            input_dir.mkdir(parents=True, exist_ok=True)
            
            # 选择视频文件
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                str(input_dir),
                "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            
            if not file_name:
                return
            
            # 创建输出目录
            current_dir = Path(__file__).parent
            video_name = Path(file_name).stem
            output_dir = current_dir / "output" / "seperate_frame" / video_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建视频检测器实例
            extractor = SketchVideoExtractor()
            
            # 显示进度对话框
            progress = QProgressDialog("正在处理视频...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("处理中")
            progress.show()
            
            # 更新进度的回调函数
            def update_progress(value, message=""):
                progress.setValue(value)
                if message:
                    progress.setLabelText(message)
                QApplication.processEvents()
            
            try:
                # 运行视频检测
                result = extractor.extract_frames(
                    str(file_name), 
                    str(output_dir)
                )
                
                # 生成缩略图
                #self.generate_thumbnails(output_dir)
                
                progress.setValue(100)
                
                if result:
                    QMessageBox.information(self, "完成", "视频检测完成！")
                    # 加载视频进行预览
                    self.load_video(file_name)
                else:
                    QMessageBox.warning(self, "警告", "视频检测可能未完全完成，请检查输出。")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"视频检测过程出错：{str(e)}")
                
            finally:
                progress.close()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化视频检测器时出错：{str(e)}")
        
    def update_preview_list(self, frames):
        """更新预览列表"""
        self.preview_list.clear()
        for frame in frames:
            item = QListWidgetItem()
            # TODO: 添加预览图
            self.preview_list.addItem(item)
        
    def generate_depth_maps(self):
        """生成深度图"""
        try:
            from depth import DepthGenerator
            
            # 创建深度图生成器实例
            generator = DepthGenerator()
            
            # 设置输入目录
            current_dir = Path(__file__).parent
            seperate_frame_dir = current_dir / "output" / "seperate_frame"
            
            if not seperate_frame_dir.exists():
                QMessageBox.warning(self, "警告", "请先运行视频检测！")
                return
            
            # 处理所有视频目录
            for video_dir in seperate_frame_dir.iterdir():
                if video_dir.is_dir():
                    print(f"\n处理视频目录: {video_dir.name}")
                    # 将 Path 对象转换为字符串
                    generator.process_video_frames(str(video_dir))
                    
            QMessageBox.information(self, "完成", "深度图生成完成！")
            
        except Exception as e:
            print(f"生成深度图时出错: {str(e)}")  # 添加详细的错误输出
            traceback.print_exc()  # 打印完整的错误堆栈
            QMessageBox.critical(self, "错误", f"生成深度图时出错：{str(e)}")

    def on_slider_changed(self, value):
        """滑块值改变时的处理函数"""
        self.show_frame(value)

    def resize_image(self, image, target_width=720, target_height=480):
        """统一的图片缩放函数"""
        if image is None:
            return None
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def show_frame(self, frame_number):
        """显示指定帧"""
        try:
            if self.video_cap is not None:
                # 更新视频预览
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.video_cap.read()
                if ret:
                    # 更新草图标注窗口
                    self.update_annotation_frame(frame_number)
                    
                    # 更新回调参数中的图像
                    if hasattr(self, 'annotator') and hasattr(self, 'keypoint_processor'):
                        self.keypoint_processor.current_image = self.annotator.image
                    
                    # 统一缩放并显示草图预览
                    frame_preview = self.resize_image(frame)
                    frame_preview = cv2.cvtColor(frame_preview, cv2.COLOR_BGR2RGB)
                    preview_image = QImage(frame_preview.data, 720, 480, 720 * 3, QImage.Format_RGB888)
                    self.sketch_preview.setPixmap(QPixmap.fromImage(preview_image))
                    
                    # 检查是否是关键帧
                    if self.detection_info and frame_number in self.keyframe_numbers:
                        keyframe_info = next(
                            (frame for frame in self.detection_info['keyframe_info'] 
                             if frame['frame_number'] == frame_number), 
                            None
                        )
                        
                        if keyframe_info:
                            # 加载并统一缩放关键帧图像
                            keyframe_path = keyframe_info['file_path']
                            keyframe = cv2.imread(keyframe_path)
                            if keyframe is not None:
                                keyframe = self.resize_image(keyframe)
                                keyframe = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
                                keyframe_image = QImage(keyframe.data, 720, 480, 720 * 3, QImage.Format_RGB888)
                                self.video_preview.setPixmap(QPixmap.fromImage(keyframe_image))
                    else:
                        self.video_preview.clear()
                        self.video_preview.setText("非关键帧")
                    
                    # 更新当前帧
                    self.current_frame = frame_number
                    self.update_frame_counter()
                    
                    # 更新Blender预览
                    blender_frame_path = self.blender_path / f"frame_{frame_number}.png"
                    if blender_frame_path.exists():
                        blender_frame = cv2.imread(str(blender_frame_path))
                        if blender_frame is not None:
                            blender_frame = self.resize_image(blender_frame)
                            blender_frame = cv2.cvtColor(blender_frame, cv2.COLOR_BGR2RGB)
                            blender_image = QImage(blender_frame.data, 720, 480, 720 * 3, QImage.Format_RGB888)
                            self.blender_preview.setPixmap(QPixmap.fromImage(blender_image))
                    else:
                        self.blender_preview.clear()
                        self.blender_preview.setText("无Blender渲染图")
                    
        except Exception as e:
            print(f"显示帧时出错：{str(e)}")

    def update_frame_counter(self):
        """更新帧计数器"""
        self.frame_counter.setText(f"帧: {self.current_frame + 1}/{self.total_frames}")

    def setup_video_controls(self):
        """设置视频控制组件"""
        # 创建视频控制容器
        self.video_control = QWidget()
        video_control_layout = QVBoxLayout(self.video_control)
        
        # 创建时间线区域
        self.timeline_widget = QWidget()
        self.timeline_widget.setFixedHeight(212)  # 与原video_preview高度相同
        self.timeline_widget.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border: 1px solid #999999;
                border-radius: 10px;
            }
        """)
        
        # 创建水平滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:horizontal {
                height: 12px;
                background: #2B2B2B;
            }
            QScrollBar::handle:horizontal {
                background: #666666;
                min-width: 20px;
                border-radius: 6px;
            }
        """)
        
        # 创建时间线内容容器
        self.timeline_content = QWidget()
        self.timeline_layout = QHBoxLayout(self.timeline_content)
        self.timeline_layout.setSpacing(2)
        self.timeline_layout.setContentsMargins(10, 10, 10, 10)
        
        self.scroll_area.setWidget(self.timeline_content)
        
        # 将滚动区域添加到时间线部件
        timeline_main_layout = QVBoxLayout(self.timeline_widget)
        timeline_main_layout.setContentsMargins(0, 0, 0, 0)
        timeline_main_layout.addWidget(self.scroll_area)
        
        # 替换原来的video_preview
        self.right_layout.replaceWidget(self.video_preview, self.timeline_widget)
        self.video_preview.hide()
        
        # 添加时间轴滑块
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.valueChanged.connect(self.on_slider_changed)
        
        # 添加帧计数器
        self.frame_counter = QLabel("0/0")
        self.frame_counter.setAlignment(Qt.AlignCenter)
        
        # 添加到布局
        video_control_layout.addWidget(self.timeline_slider)
        video_control_layout.addWidget(self.frame_counter)
        
        # 将控制组件添加到视频预览区域下方
        self.right_layout.insertWidget(2, self.video_control)

    def update_timeline(self):
        """更新时间线显示"""
        # 清除现有的缩略图
        for i in reversed(range(self.timeline_layout.count())):
            self.timeline_layout.itemAt(i).widget().deleteLater()
        
        if not self.detection_info or not self.keyframe_numbers:
            return
        
        # 缩略图大小
        THUMB_WIDTH = 160
        THUMB_HEIGHT = 90
        
        # 为每个关键帧创建缩略图
        for frame_info in self.detection_info['keyframe_info']:
            # 创建缩略图容器
            thumb_container = QWidget()
            thumb_container.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT + 20)
            thumb_container.setStyleSheet("""
                QWidget {
                    background-color: transparent;
                }
                QWidget:hover {
                    background-color: rgba(74, 144, 226, 0.3);
                    border-radius: 5px;
                }
            """)
            
            # 创建缩略图标签
            thumb_label = QLabel()
            thumb_label.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT)
            
            # 加载并缩放图像
            frame_path = frame_info['file_path']
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame.data, frame.shape[1], frame.shape[0], 
                             frame.shape[1] * 3, QImage.Format_RGB888)
                thumb_label.setPixmap(QPixmap.fromImage(image))
            
            # 创建帧号标签
            frame_number = QLabel(f"Frame {frame_info['frame_number']}")
            frame_number.setAlignment(Qt.AlignCenter)
            frame_number.setStyleSheet("color: white;")
            
            # 布局
            layout = QVBoxLayout(thumb_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            layout.addWidget(thumb_label)
            layout.addWidget(frame_number)
            
            # 创建点击事件处理器
            thumb_container.mousePressEvent = lambda e, frame_num=frame_info['frame_number']: self.on_thumbnail_clicked(frame_num)
            
            # 添加到时间线
            self.timeline_layout.addWidget(thumb_container)
        
        # 添加弹性空间
        self.timeline_layout.addStretch()

    def on_thumbnail_clicked(self, frame_number):
        """处理缩略图点击事件"""
        try:
            # 更新滑块位置
            self.timeline_slider.setValue(frame_number)
            
            # 显示对应帧
            self.show_frame(frame_number)
            
            print(f"跳转到第 {frame_number} 帧")
            
        except Exception as e:
            print(f"跳转帧时出错：{str(e)}")

    def load_video_file(self):
        """加载视频文件"""
        try:
            # 设置默认目录为项目的input文件夹
            current_dir = Path(__file__).parent
            input_dir = current_dir / "input" / "video"
            
            # 确保input目录存在
            input_dir.mkdir(parents=True, exist_ok=True)
            
            # 选择视频文件
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                str(input_dir),  # 使用input目录作为默认路径
                "视频文件 (*.mp4 *.avi *.mov *.mkv)"
            )
            
            if file_name:
                self.load_video(file_name)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载视频时出错：{str(e)}")

    def load_video(self, file_name):
        """加载视频"""
        try:
            # 确保file_name是字符串类型
            file_name = str(file_name)
            
            # 关闭之前的视频
            if hasattr(self, 'video_cap') and self.video_cap is not None:
                self.video_cap.release()
            
            # 检查文件是否存在
            if not Path(file_name).exists():
                raise FileNotFoundError(f"视频文件不存在: {file_name}")
            
            # 初始化视频捕获
            self.video_cap = cv2.VideoCapture(file_name)
            if not self.video_cap.isOpened():
                raise ValueError(f"无法打开视频文件: {file_name}")
            
            # 保存视频路径
            self.video_path = file_name
            
            # 获取视频信息
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            print(f"成功加载视频:")
            print(f"- 路径: {self.video_path}")
            print(f"- 总帧数: {self.total_frames}")
            print(f"- FPS: {self.fps}")
            
            # 加载检测信息
            self.load_detection_info(file_name)
            
            # 更新UI
            self.update_timeline()
            self.timeline_slider.setMaximum(self.total_frames - 1)
            self.update_frame_counter()
            self.show_frame(0)
            
        except Exception as e:
            logger.error(f"加载视频时出错: {str(e)}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            QMessageBox.critical(self, "错误", f"加载视频时出错：{str(e)}")

    def load_detection_info(self, video_path):
        """加载检测信息"""
        try:
            # 确保使用正确的路径分隔符
            video_path = Path(video_path)
            video_name = video_path.stem
            
            # 构建detection_info.json的路径
            current_dir = Path(__file__).parent
            detection_info_path = current_dir / "output" / "seperate_frame" / video_name / "detection_info.json"
            
            logger.info(f"尝试加载检测信息: {detection_info_path}")
            
            if not detection_info_path.exists():
                logger.warning(f"未找到检测信息文件：{detection_info_path}")
                self.detection_info = None
                self.keyframe_numbers = []
                return
            
            # 加载检测信息
            with open(detection_info_path, 'r', encoding='utf-8') as f:
                self.detection_info = json.load(f)
            
            # 提取关键帧帧号
            self.keyframe_numbers = [
                frame['frame_number'] 
                for frame in self.detection_info.get('keyframe_info', [])
            ]
            
            logger.info(f"加载了 {len(self.keyframe_numbers)} 个关键帧信息")
            
        except Exception as e:
            logger.error(f"加载检测信息时出错：{str(e)}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            self.detection_info = None
            self.keyframe_numbers = []
        
    def open_audio_generator(self):
        """打开音频生成器窗口"""
        try:
            from audio_generate import AudioVideoEditor
        
            if not hasattr(self, 'audio_generator'):
                self.audio_generator = AudioVideoEditor()
        
        # 传递视频相关信息
            self.audio_generator.video_cap = self.video_cap
            self.audio_generator.total_frames = self.total_frames
            self.audio_generator.fps = self.fps
        
        # 如果有关键帧信息，更新缩略图
            if hasattr(self, 'detection_info') and self.detection_info:
                self.audio_generator.update_thumbnails(self.detection_info['keyframe_info'])
            self.audio_generator.set_video_info(self.video_path, Path(self.video_path).stem)
        # 显示当前帧
            if self.current_frame:
                self.audio_generator.show_frame(self.current_frame)
            
            self.audio_generator.show()
        
        except Exception as e:
            logger.error(f"打开音频生成器时出错：{str(e)}")
            QMessageBox.critical(self, "错误", f"打开音频生成器时出错：{str(e)}")  

    def closeEvent(self, event):
        """重写关闭事件处理"""
        try:
            # 停止草图标注循环
            if hasattr(self, 'annotation_running'):
                self.annotation_running = False
            
            # 关闭所有OpenCV窗口
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # 调用父类的关闭事件
            super().closeEvent(event)
        except Exception as e:
            logger.error(f"关闭窗口时出错: {str(e)}")

    def update_blender_timeline(self):
        """更新Blender时间线显示"""
        try:
            # 清除现有的缩略图
            for i in reversed(range(self.blender_timeline_layout.count())):
                self.blender_timeline_layout.itemAt(i).widget().deleteLater()
            
            # 检查Blender路径是否存在
            if not self.blender_path.exists():
                print(f"Blender路径不存在: {self.blender_path}")
                return
            
            # 获取所有blender渲染图片并排序
            blender_images = sorted(
                self.blender_path.glob("frame_*.png"),
                key=lambda x: int(x.stem.split('_')[1])  # 从"frame_1"提取数字1
            )
            
            if not blender_images:
                print(f"未找到Blender渲染图: {self.blender_path}")
                return
            
            print(f"找到 {len(blender_images)} 个Blender渲染图")
            
            # 缩略图大小
            THUMB_WIDTH = 160
            THUMB_HEIGHT = 90
            
            # 为每个图片创建缩略图
            for img_path in blender_images:
                try:
                    # 获取帧号
                    frame_num = int(img_path.stem.split('_')[1])
                    
                    # 创建缩略图容器
                    thumb_container = QWidget()
                    thumb_container.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT + 20)
                    thumb_container.setStyleSheet("""
                        QWidget {
                            background-color: transparent;
                        }
                        QWidget:hover {
                            background-color: rgba(74, 144, 226, 0.3);
                            border-radius: 5px;
                        }
                    """)
                    
                    # 创建缩略图标签
                    thumb_label = QLabel()
                    thumb_label.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT)
                    thumb_label.setAlignment(Qt.AlignCenter)
                    
                    # 加载并缩放图像
                    frame = cv2.imread(str(img_path))
                    if frame is not None:
                        # 先缩放到720x480，再缩放到缩略图大小
                        frame = self.resize_image(frame)  # 统一缩放到720x480
                        frame = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT))  # 缩放到缩略图大小
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = QImage(frame.data, frame.shape[1], frame.shape[0], 
                                     frame.shape[1] * 3, QImage.Format_RGB888)
                        thumb_label.setPixmap(QPixmap.fromImage(image))
                    else:
                        print(f"无法加载图像: {img_path}")
                        continue
                    
                    # 创建帧号标签
                    frame_number = QLabel(f"Frame {frame_num}")
                    frame_number.setAlignment(Qt.AlignCenter)
                    frame_number.setStyleSheet("color: white;")
                    
                    # 布局
                    layout = QVBoxLayout(thumb_container)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(2)
                    layout.addWidget(thumb_label)
                    layout.addWidget(frame_number)
                    
                    # 添加点击事件
                    thumb_container.mousePressEvent = lambda e, fn=frame_num: self.on_thumbnail_clicked(fn)
                    
                    # 添加到时间线
                    self.blender_timeline_layout.addWidget(thumb_container)
                    
                except Exception as e:
                    print(f"处理图像时出错 {img_path}: {str(e)}")
                    continue
            
            # 添加弹性空间
            self.blender_timeline_layout.addStretch()
            
        except Exception as e:
            print(f"更新Blender时间线时出错：{str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoEditor()
    window.show()
    sys.exit(app.exec())