import sys
import os
import subprocess
import threading
import signal
import time
import re
import webbrowser

try:
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QLabel, QLineEdit, QCheckBox,
                                 QTextEdit, QGroupBox, QFileDialog, QMessageBox, 
                                 QComboBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSettings
except ImportError:
    print("❌ 缺少必要的依赖库！请先运行: pip install PyQt5")
    sys.exit(1)

class LogSignal(QObject):
    update = pyqtSignal(str)

class IsaacDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Isaac Lab 强化学习控制台")
        self.resize(1000, 750) 
        
        self.train_process = None
        self.play_process = None 
        self.tb_process = None
        self.work_dir = "/home/ubuntu/isaaclab_files"
        self.current_task_id = "Isaac-Velocity-Flat-Robot2-v0" 
        
        # 初始化设置管理器
        self.settings = QSettings("IsaacLabTools", "Dashboard")
        
        self.log_signal = LogSignal()
        self.log_signal.update.connect(self.append_log)

        self.init_ui()
        # 确保在填充完环境后再加载保存的状态
        self.load_saved_settings()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. 任务路径与解析区
        task_group = QGroupBox("📂 任务选择与自动解析")
        task_layout = QVBoxLayout()
        
        path_btn_layout = QHBoxLayout()
        self.btn_select_path = QPushButton("📁 选择任务目录")
        self.btn_select_path.setStyleSheet("padding: 5px; font-weight: bold;")
        self.btn_select_path.clicked.connect(self.select_task_directory)
        path_btn_layout.addWidget(self.btn_select_path)
        
        task_layout.addLayout(path_btn_layout)
        
        self.path_display = QLineEdit("未选择目录，将使用默认 Task ID")
        self.path_display.setReadOnly(True)
        self.path_display.setStyleSheet("background-color: #f0f0f0; color: #666;")
        task_layout.addWidget(self.path_display)
        
        self.id_display_label = QLabel(f"当前识别到的 Task: <b>{self.current_task_id}</b>")
        self.id_display_label.setStyleSheet("color: #0055ff; font-size: 14px;")
        task_layout.addWidget(self.id_display_label)
        
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)

        # 2. 环境与训练参数配置区
        config_group = QGroupBox("⚙️ 环境与参数配置")
        config_layout = QHBoxLayout()
        
        config_layout.addWidget(QLabel("Conda环境:"))
        self.conda_env_combo = QComboBox()
        self.conda_env_combo.setFixedWidth(150)
        self.conda_env_combo.setEditable(True) 
        self.conda_env_combo.setStyleSheet("background-color: #f0f0f0; color: #666;")
        
        # 扫描并添加环境
        self.populate_conda_envs() 
        
        # 当用户手动输入或选择改变时，实时保存
        self.conda_env_combo.currentTextChanged.connect(self.save_current_env)
        
        config_layout.addWidget(self.conda_env_combo)
        
        config_layout.addSpacing(15)
        config_layout.addWidget(QLabel("环境数量:"))
        self.num_envs_input = QLineEdit("8192")
        self.num_envs_input.setFixedWidth(80)
        config_layout.addWidget(self.num_envs_input)

        # ✨ 新增：训练次数UI及实时保存逻辑
        config_layout.addSpacing(15)
        config_layout.addWidget(QLabel("训练次数:"))
        self.max_iters_input = QLineEdit("1500")
        self.max_iters_input.setFixedWidth(80)
        self.max_iters_input.textChanged.connect(self.save_max_iters)
        config_layout.addWidget(self.max_iters_input)
        
        config_layout.addSpacing(15)
        self.headless_check = QCheckBox("无头模式 (--headless)")
        self.headless_check.setChecked(True)
        config_layout.addWidget(self.headless_check)
        config_layout.addStretch()
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # 3. 按钮控制区
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶️ 开始训练")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_training)
        
        self.btn_stop = QPushButton("⏹️ 停止训练")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        
        self.btn_play = QPushButton("开启可视化")
        self.btn_play.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.btn_play.clicked.connect(self.start_play)
        
        self.btn_stop_play = QPushButton("关闭可视化")
        self.btn_stop_play.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop_play.setEnabled(False)
        self.btn_stop_play.clicked.connect(self.stop_play)

        self.btn_kill_all = QPushButton("强制清理进程")
        self.btn_kill_all.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 10px;")
        self.btn_kill_all.clicked.connect(self.kill_all_python)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addSpacing(20) 
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop_play)
        btn_layout.addSpacing(20) 
        btn_layout.addWidget(self.btn_kill_all)
        
        main_layout.addLayout(btn_layout)

        # 4. 日志区
        log_group = QGroupBox("实时终端日志")
        log_layout = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: Consolas;")
        log_layout.addWidget(self.log_area)
        log_group.setLayout(log_layout)
        
        main_layout.addWidget(log_group, stretch=1)

    # ================= 核心逻辑 =================

    def populate_conda_envs(self):
        """扫描 Conda 环境"""
        envs = []
        try:
            # 尝试获取环境列表
            process = subprocess.Popen(["conda", "env", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = process.communicate()
            
            for line in out.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts:
                    env_name = parts[0]
                    # 排除路径环境，只取名字
                    if not os.path.isabs(env_name):
                        envs.append(env_name)
        except Exception as e:
            print(f"扫描 Conda 失败: {e}")
            
        if not envs:
            envs = ["base", "isaaclab"]
            
        self.conda_env_combo.clear()
        self.conda_env_combo.addItems(envs)

    def save_current_env(self, text):
        """实时保存环境配置"""
        if text.strip():
            self.settings.setValue("conda_env", text.strip())

    # ✨ 新增：实时保存训练次数
    def save_max_iters(self, text):
        if text.strip():
            self.settings.setValue("max_iterations", text.strip())

    def load_saved_settings(self):
        """加载历史记录"""
        # 加载路径
        saved_path = self.settings.value("last_task_path", "")
        if saved_path and os.path.exists(saved_path):
            self.process_task_directory(saved_path)
            
        # 加载环境：从配置文件读取上一次的值，并设为当前显示
        saved_conda = self.settings.value("conda_env", "isaaclab")
        self.conda_env_combo.setCurrentText(saved_conda)

        # ✨ 新增：加载上一次的训练次数，默认1500
        saved_iters = self.settings.value("max_iterations", "1500")
        self.max_iters_input.setText(str(saved_iters))

    def select_task_directory(self):
        initial_dir = self.settings.value("last_task_path", self.work_dir)
        dir_path = QFileDialog.getExistingDirectory(self, "选择任务所在文件夹", initial_dir)
        if dir_path:
            self.process_task_directory(dir_path)

    def process_task_directory(self, dir_path):
        self.path_display.setText(dir_path)
        init_file = os.path.join(dir_path, "__init__.py")

        if not os.path.exists(init_file):
            return

        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'(?:id=|register\()\s*["\'](.*?)["\']', content)
                if match:
                    self.current_task_id = match.group(1)
                    self.id_display_label.setText(f"当前识别到的 Task: <b>{self.current_task_id}</b>")
                    self.settings.setValue("last_task_path", dir_path)
        except:
            pass

    def append_log(self, text):
        self.log_area.append(text)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def read_output(self, process):
        while True:
            line = process.stdout.readline()
            if not line: break
            self.log_signal.update.emit(line.strip())
        process.stdout.close()

    def run_command_in_bg(self, cmd_list):
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        env["PYTHONUNBUFFERED"] = "1" 
        
        process = subprocess.Popen(cmd_list, cwd=self.work_dir, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env,
                                   bufsize=1, preexec_fn=os.setsid)
        threading.Thread(target=self.read_output, args=(process,), daemon=True).start()
        return process

    def wrap_conda_cmd(self, base_cmd):
        conda_env = self.conda_env_combo.currentText().strip()
        # 冗余保存一次，确保万无一失
        self.save_current_env(conda_env)
        
        if conda_env:
            return ["conda", "run", "-n", conda_env, "--no-capture-output"] + base_cmd
        return base_cmd

    def kill_all_python(self):
        reply = QMessageBox.warning(self, '⚠️ 确认', '执行 killall -9 python 吗？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            os.system("killall -9 python")

    def start_training(self):
        envs = self.num_envs_input.text()
        base_cmd = ["python", "scripts/reinforcement_learning/rsl_rl/train.py", "--task", self.current_task_id, "--num_envs", envs]
        
        # ✨ 新增：读取并附加最大训练次数参数 (--max_iterations)
        max_iters = self.max_iters_input.text().strip()
        if max_iters.isdigit():
            base_cmd.extend(["--max_iterations", max_iters])

        if self.headless_check.isChecked(): base_cmd.append("--headless")
        
        cmd = self.wrap_conda_cmd(base_cmd)
        self.append_log(f"\n🚀 [系统] 开始训练: {' '.join(cmd)}")
        self.train_process = self.run_command_in_bg(cmd)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        if not self.tb_process:
            tb_cmd = self.wrap_conda_cmd(["tensorboard", "--logdir", "logs/rsl_rl", "--port", "6006"])
            self.tb_process = self.run_command_in_bg(tb_cmd)
        QTimer.singleShot(2000, lambda: webbrowser.open("http://localhost:6006"))

    def stop_training(self):
        if self.train_process:
            try: os.killpg(os.getpgid(self.train_process.pid), signal.SIGTERM)
            except: pass
            self.train_process = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def start_play(self):
        base_cmd = ["python", "scripts/reinforcement_learning/rsl_rl/play.py", "--task", self.current_task_id, "--num_envs", "64"]
        cmd = self.wrap_conda_cmd(base_cmd)
        self.play_process = self.run_command_in_bg(cmd)
        self.btn_play.setEnabled(False)
        self.btn_stop_play.setEnabled(True)

    def stop_play(self):
        if self.play_process:
            try: os.killpg(os.getpgid(self.play_process.pid), signal.SIGTERM)
            except: pass
            self.play_process = None
        self.btn_play.setEnabled(True)
        self.btn_stop_play.setEnabled(False)

    # ---------------- 关键改进：关闭事件 ----------------

    def closeEvent(self, event):
        """当用户关闭窗口时执行"""
        # 1. 强制保存当前的 Conda 环境名和任务路径
        current_env = self.conda_env_combo.currentText().strip()
        self.settings.setValue("conda_env", current_env)
        self.settings.setValue("last_task_path", self.path_display.text())
        
        # ✨ 新增：在关闭时确保保存训练次数
        self.settings.setValue("max_iterations", self.max_iters_input.text().strip())

        # 2. 停止所有后台进程
        self.stop_training()
        self.stop_play()
        if self.tb_process:
            try: os.killpg(os.getpgid(self.tb_process.pid), signal.SIGTERM)
            except: pass
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = IsaacDashboard()
    dashboard.show()
    sys.exit(app.exec_())