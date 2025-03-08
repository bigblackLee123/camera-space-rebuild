import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import cv2
import numpy as np
from pathlib import Path
import json
import requests
import traceback
import time
import shutil
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

class AudioVideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音频生成器")
        self.setFixedSize(1080, 720)  # 调整窗口大小
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QLabel {
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #3D3D3D;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
            QTextEdit {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #3D3D3D;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        
        # 主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. 视频播放区域
        self.setup_video_player()
        
        # 2. 缩略图时间线
        self.setup_timeline()
        
        # 3. 音频生成区域
        self.setup_audio_generator()
        
        # 初始化音频播放器
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        # 初始化视频和音频状态
        self.current_frame = 0
        self.is_playing = False
        self.audio_file = None
        self.audio_start_frame = None
        self.audio_end_frame = None

    def setup_video_player(self):
        """设置视频播放区域"""
        # 视频显示
        self.video_display = ClickableLabel()  # 自定义可点击Label
        self.video_display.setFixedSize(600, 400)
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #2B2B2B;
                border: 1px solid #3D3D3D;
                border-radius: 10px;
            }
            QLabel:hover {
                border: 1px solid #4D4D4D;
            }
        """)
        self.video_display.clicked.connect(self.toggle_play)  # 连接点击事件
        
        self.layout.addWidget(self.video_display, alignment=Qt.AlignCenter)

    def setup_timeline(self):
        """设置缩略图时间线"""
        # 创建一个固定宽度的容器来包裹时间线
        timeline_container = QWidget()
        timeline_container.setFixedWidth(720)  # 与视频显示区域同宽
        container_layout = QVBoxLayout(timeline_container)
        container_layout.setContentsMargins(0, 30, 0, 0)
        
        # 时间线主体
        self.timeline_widget = QWidget()
        self.timeline_widget.setFixedSize(720, 80)
        self.timeline_widget.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border: 1px solid #3D3D3D;
                border-radius: 10px;
            }
        """)
        
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
                height: 12px;  /* 增加滚动条高度 */
                background: #2B2B2B;
                margin: 0px 10px 0px 10px;  /* 增加左右边距 */
            }
            QScrollBar::handle:horizontal {
                background: #4D4D4D;
                min-width: 20px;
                border-radius: 6px;  /* 增加圆角 */
                margin: 2px 0px 2px 0px;  /* 增加上下边距使滚动条更居中 */
            }
            QScrollBar::handle:horizontal:hover {
                background: #666666;  /* 鼠标悬停时颜色变化 */
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        self.timeline_content = QWidget()
        self.timeline_layout = QHBoxLayout(self.timeline_content)
        self.timeline_layout.setSpacing(4)
        self.timeline_layout.setContentsMargins(10, 10, 10, 10)
        
        self.scroll_area.setWidget(self.timeline_content)
        
        timeline_main_layout = QVBoxLayout(self.timeline_widget)
        timeline_main_layout.setContentsMargins(0, 0, 0, 0)
        timeline_main_layout.addWidget(self.scroll_area)
        
        # 添加选区按钮
        self.select_btn = QPushButton("选择区间")
        self.select_btn.setFixedSize(100, 30)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #4D4D4D;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:checked {
                background-color: #2D8CF0;
            }
        """)
        self.select_btn.setCheckable(True)  # 使按钮可切换
        self.select_btn.clicked.connect(self.toggle_selection_mode)
        
        # 将按钮添加到时间线容器布局
        container_layout.addWidget(self.select_btn, alignment=Qt.AlignLeft)
        container_layout.addWidget(self.timeline_widget)
        
        # 将容器添加到主布局，并设置居中对齐
        self.layout.addWidget(timeline_container, alignment=Qt.AlignCenter)

    def setup_audio_generator(self):
        """设置音频生成区域"""
        # 创建固定宽度的容器
        audio_container = QWidget()
        audio_container.setFixedWidth(720)  # 与视频和时间线容器同宽
        audio_container.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border: 1px solid #3D3D3D;
                border-radius: 10px;
            }
        """)
        
        # 创建主布局
        main_layout = QVBoxLayout(audio_container)
        main_layout.setContentsMargins(20, 15, 20, 15)  # 设置内边距
        
        # 创建水平布局用于放置输入框和按钮
        audio_layout = QHBoxLayout()
        audio_layout.setSpacing(15)
        
        # 提示文本输入框
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("输入音频生成提示词...")
        self.prompt_input.setFixedSize(540, 80)  # 固定宽度为540
        self.prompt_input.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3D3D3D;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        
        # 右侧控制区
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setSpacing(10)
        
        self.duration_label = QLabel("选中片段时长: 0秒")
        self.duration_label.setStyleSheet("color: #CCCCCC; font-size: 13px;")
        
        self.generate_btn = QPushButton("生成音频")
        self.generate_btn.setFixedSize(120, 40)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4D4D4D;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_audio)
        
        control_layout.addWidget(self.duration_label)
        control_layout.addWidget(self.generate_btn)
        control_layout.addStretch()
        
        # 添加到水平布局
        audio_layout.addWidget(self.prompt_input)
        audio_layout.addWidget(control_container)
        
        # 将水平布局添加到主布局
        main_layout.addLayout(audio_layout)
        
        # 创建一个容器来包裹音频生成区域
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 30, 0, 0)  # 增加顶部边距
        container_layout.addWidget(audio_container)
        
        # 将容器添加到主布局，并设置居中对齐
        self.layout.addWidget(container, alignment=Qt.AlignCenter)

    def show_frame(self, frame_number):
        """显示指定帧"""
        try:
            if self.video_cap is None:
                return
                
            # 更新当前帧号
            self.current_frame = frame_number
            
            # 设置视频帧位置
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video_cap.read()
            
            if ret:
                # 转换颜色空间
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整图像大小以适应显示区域
                display_size = self.video_display.size()
                frame = cv2.resize(frame, (display_size.width(), display_size.height()))
                
                # 转换为Qt图像
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # 显示图像
                self.video_display.setPixmap(QPixmap.fromImage(qt_image))
                
        except Exception as e:
            print(f"显示帧时出错: {str(e)}")

    def toggle_play(self):
        """切换播放/暂停状态"""
        try:
            self.is_playing = not self.is_playing
            
            if self.is_playing:
                # 创建定时器用于播放
                if not hasattr(self, 'play_timer'):
                    self.play_timer = QTimer()
                    self.play_timer.timeout.connect(self.play_next_frame)
                
                # 开始播放
                self.play_timer.start(33)  # ~30fps
                
                # 如果在音频范围内，播放音频
                if (self.audio_file and self.audio_start_frame and 
                    self.audio_start_frame <= self.current_frame <= self.audio_end_frame):
                    self.media_player.play()
            else:
                # 停止播放
                if hasattr(self, 'play_timer'):
                    self.play_timer.stop()
                if self.media_player:
                    self.media_player.pause()
                    
        except Exception as e:
            print(f"切换播放状态时出错: {str(e)}")

    def play_next_frame(self):
        """播放下一帧"""
        try:
            if self.video_cap is None:
                return
            
            # 获取当前帧位置
            self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 如果到达视频末尾，停止播放
            if self.current_frame >= self.total_frames:
                self.play_timer.stop()
                self.is_playing = False
                if self.media_player:
                    self.media_player.stop()
                return
            
            # 读取并显示下一帧
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_size = self.video_display.size()
                frame = cv2.resize(frame, (display_size.width(), display_size.height()))
                
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_display.setPixmap(QPixmap.fromImage(qt_image))
                
                # 检查并播放音频
                if (self.audio_file and self.audio_start_frame and 
                    self.audio_start_frame <= self.current_frame <= self.audio_end_frame):
                    
                    # 如果音频未在播放，开始播放
                    if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                        print("Starting audio playback")
                        self.media_player.play()
                    
                    # 同步音频位置
                    total_frames = self.audio_end_frame - self.audio_start_frame
                    current_audio_frame = self.current_frame - self.audio_start_frame
                    position = (current_audio_frame / total_frames) * self.media_player.duration()
                    self.media_player.setPosition(int(position))
                    
                    print(f"Current frame: {self.current_frame}, Audio position: {position}")
                
        except Exception as e:
            print(f"播放下一帧时出错: {str(e)}")
            traceback.print_exc()
            if hasattr(self, 'play_timer'):
                self.play_timer.stop()
                self.is_playing = False

    def toggle_selection_mode(self):
        """切换选区模式"""
        self.selection_mode = self.select_btn.isChecked()
        if self.selection_mode:
            self.start_frame = None
            self.end_frame = None
            self.select_btn.setText("选择中...")
        else:
            self.select_btn.setText("选择区间")

    def on_thumbnail_clicked(self, frame_number, duration):
        """处理缩略图点击事件"""
        try:
            if hasattr(self, 'selection_mode') and self.selection_mode:
                if self.start_frame is None:
                    # 记录起始帧
                    self.start_frame = frame_number
                    self.select_btn.setText("选择结束帧...")
                else:
                    # 记录结束帧
                    self.end_frame = frame_number
                    self.calculate_duration()
                    self.selection_mode = False
                    self.select_btn.setChecked(False)
                    self.select_btn.setText("选择区间")
            else:
                # 普通点击显示帧
                self.show_frame(frame_number)
            
        except Exception as e:
            print(f"选择缩略图时出错：{str(e)}")

    def calculate_duration(self):
        """计算选区时长并更新"""
        if self.start_frame is not None and self.end_frame is not None:
            # 确保开始帧在前，结束帧在后
            start = min(self.start_frame, self.end_frame)
            end = max(self.start_frame, self.end_frame)
            
            # 计算时长（秒）
            frame_count = end - start
            duration = frame_count / self.fps if self.fps > 0 else frame_count / 30
            
            # 更新显示
            self.selected_duration = duration
            self.duration_label.setText(f"选中片段时长: {duration:.2f}秒")
            
            # 更新工作流API中的时长
            self.update_workflow_duration(duration)

    def update_workflow_duration(self, duration):
        """更新工作流API中的时长"""
        try:
            workflow_api = self.load_workflow_api()
            if workflow_api:
                workflow_api["6"]["inputs"]["seconds"] = duration
                
                # 保存更新后的工作流
                with open("workflow_api.json", 'w', encoding='utf-8') as f:
                    json.dump(workflow_api, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"更新工作流时长时出错: {str(e)}")

    def check_comfyui_status(self):
        """检查ComfyUI服务状态"""
        try:
            response = requests.get("http://127.0.0.1:8188/", timeout=5)
            return response.status_code == 200
        except:
            return False

    def show_success_message(self, title, message):
        """显示成功消息"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        
        # 设置消息框样式
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QMessageBox QLabel {
                color: #000000;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #4D4D4D;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 13px;
            }
            QMessageBox QPushButton:hover {
                background-color: #666666;
            }
        """)
        
        msg.exec_()

    def generate_audio(self):
        """生成音频"""
        if not hasattr(self, 'selected_duration') or not hasattr(self, 'start_frame') or not hasattr(self, 'end_frame'):
            self.show_error_message("警告", "请先选择时间区间!")
            return
        
        prompt = self.prompt_input.toPlainText()
        if not prompt:
            self.show_error_message("警告", "请输入提示词!")
            return
        
        try:
            workflow_api = self.load_workflow_api()
            if not workflow_api:
                self.show_error_message("错误", "无法加载workflow_api.json文件")
                return
            
            # 使用ComfyUI默认的保存路径
            workflow_api["15"]["inputs"]["filename_prefix"] = "audio/ComfyUI"
            
            # 准备目标路径
            target_dir = Path("output/seperate_frame") / self.video_name / "audio"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_filename = f"frame{self.start_frame}_to_frame{self.end_frame}.flac"
            target_path = target_dir / target_filename
            
            # 发送请求到ComfyUI
            print("Sending request to ComfyUI...")
            response = requests.post(
                "http://127.0.0.1:8188/prompt",
                json={"prompt": workflow_api},
                timeout=10
            )
            
            if response.status_code == 200:
                # 使用硬编码的ComfyUI输出路径
                comfyui_output = Path("F:/ai_program_2/ComdyUI/ComfyUI_windows_portable/ComfyUI/output/audio")
                
                # 等待文件生成（最多等待30秒）
                source_file = None
                for _ in range(30):
                    # 查找ComfyUI*.flac文件（改为.flac）
                    audio_files = list(comfyui_output.glob("ComfyUI*.flac"))
                    if audio_files:
                        source_file = max(audio_files, key=lambda x: x.stat().st_mtime)
                        print(f"Found audio file: {source_file}")
                        break
                    print("Waiting for audio file...")
                    time.sleep(1)
                
                if source_file and source_file.exists():
                    try:
                        # 使用.flac扩展名
                        target_filename = f"frame{self.start_frame}_to_frame{self.end_frame}.flac"
                        target_path = target_dir / target_filename
                        
                        # 移动文件
                        shutil.move(str(source_file), str(target_path))
                        print(f"Moved file from {source_file} to {target_path}")
                        
                        # 加载音频
                        self.load_audio(self.start_frame, self.end_frame)
                        
                        self.show_success_message("成功", 
                            f"音频已生成并移动到目标位置!\n"
                            f"保存路径: {target_path}")
                    except Exception as e:
                        self.show_error_message("错误", 
                            f"移动音频文件时出错:\n{str(e)}\n"
                            f"源文件: {source_file}\n"
                            f"目标路径: {target_path}")
                else:
                    self.show_error_message("错误", "未找到生成的音频文件，请检查ComfyUI输出目录")
            else:
                self.show_error_message("错误", f"API请求失败: {response.status_code}\n响应内容: {response.text}")
                
        except Exception as e:
            self.show_error_message("错误", f"生成音频时出错:\n{str(e)}")
            print(f"Detailed error: {traceback.format_exc()}")

    def load_workflow_api(self):
        """加载工作流API配置"""
        try:
            with open("workflow_api.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载API配置失败: {str(e)}")
            return None

    def update_thumbnails(self, keyframe_info):
        """更新缩略图显示"""
        # 清除现有的缩略图
        for i in reversed(range(self.timeline_layout.count())):
            self.timeline_layout.itemAt(i).widget().deleteLater()
        
        # 缩略图大小
        THUMB_WIDTH = 100
        THUMB_HEIGHT = 60
        
        # 为每个关键帧创建缩略图
        for frame_info in keyframe_info:
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
            
            # 计算该帧的时长（基于fps）
            if hasattr(self, 'fps') and self.fps > 0:
                duration = 1.0 / self.fps  # 每帧的时长（秒）
            else:
                duration = 0.033  # 默认30fps
                
            # 创建点击事件处理器
            thumb_container.mousePressEvent = lambda e, fn=frame_info['frame_number'], d=duration: self.on_thumbnail_clicked(fn, d)
            
            # 添加到时间线
            self.timeline_layout.addWidget(thumb_container)
        
        # 添加弹性空间
        self.timeline_layout.addStretch()

    def set_video_info(self, video_path, video_name):
        """设置视频信息"""
        self.video_path = video_path
        self.video_name = video_name
        
        # 打开视频文件
        self.video_cap = cv2.VideoCapture(video_path)
        if self.video_cap.isOpened():
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)

    def show_error_message(self, title, message):
        """显示错误消息"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        
        # 设置消息框样式
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QMessageBox QLabel {
                color: #000000;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #4D4D4D;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 13px;
            }
            QMessageBox QPushButton:hover {
                background-color: #666666;
            }
        """)
        
        msg.exec_()

    def load_audio(self, start_frame, end_frame):
        """加载指定帧范围的音频"""
        try:
            # 构建音频文件路径（使用.flac扩展名）
            audio_path = Path("output/seperate_frame") / self.video_name / "audio" / f"frame{start_frame}_to_frame{end_frame}.flac"
            
            if not audio_path.exists():
                print(f"音频文件不存在: {audio_path}")
                return False
            
            print(f"Loading audio file: {audio_path}")
            
            # 重置并重新初始化音频播放器
            if hasattr(self, 'media_player'):
                self.media_player.stop()
            
            # 设置音频文件
            self.audio_file = str(audio_path.absolute())  # 使用绝对路径
            self.audio_start_frame = start_frame
            self.audio_end_frame = end_frame
            
            # 加载音频文件
            self.media_player.setSource(QUrl.fromLocalFile(self.audio_file))
            self.audio_output.setVolume(1.0)  # 确保音量设置正确
            
            print(f"Audio loaded: {self.audio_file}")
            print(f"Audio duration: {self.media_player.duration()}")
            print(f"Audio state: {self.media_player.playbackState()}")
            
            return True
            
        except Exception as e:
            print(f"加载音频时出错: {str(e)}")
            traceback.print_exc()  # 打印详细错误信息
            return False

class ClickableLabel(QLabel):
    """自定义可点击的Label类"""
    clicked = Signal()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioVideoEditor()
    window.show()
    sys.exit(app.exec())