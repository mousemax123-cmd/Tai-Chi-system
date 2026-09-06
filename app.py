import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import time
from data_process import extract_landmarks
from scoring import visualize_comparison
from improved_scoring import (
    score_movement_improved, 
    get_error_suggestions_improved,
    initialize_scoring_calibration,
    score_movement_improved_with_calibration
)
from pose_visualization import create_pose_comparison_visualization

# 设置页面
st.set_page_config(page_title="太极拳动作评分系统", layout="wide")

# 初始化MediaPipe
mp_pose = mp.solutions.pose
# 使用更高精度的参数设置
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # 使用最复杂的模型以提高精度
    smooth_landmarks=True,  # 启用关键点平滑
    enable_segmentation=False,  # 关闭分割以提高性能
    smooth_segmentation=False,
    min_detection_confidence=0.7,  # 提高检测置信度阈值
    min_tracking_confidence=0.7   # 提高跟踪置信度阈值
)
mp_drawing = mp.solutions.drawing_utils

# 创建必要的文件夹
def create_folders():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 创建标准动作和学生动作文件夹
    standard_dir = os.path.join(data_dir, "standard")
    student_dir = os.path.join(data_dir, "student")
    processed_dir = os.path.join(data_dir, "processed")
    results_dir = os.path.join(base_dir, "results")
    
    if not os.path.exists(standard_dir):
        os.makedirs(standard_dir)
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "standard_dir": standard_dir,
        "student_dir": student_dir,
        "processed_dir": processed_dir,
        "results_dir": results_dir
    }

# 创建文件夹
folders = create_folders()

# 初始化评分校准系统
@st.cache_resource
def initialize_calibration():
    """初始化评分校准系统"""
    # 使用默认的校准参数，适合教学场景
    # 目标平均分75分，标准差12分，符合正态分布
    calibrator, success = initialize_scoring_calibration(target_mean=75.0, target_std=12.0)
    if success:
        st.sidebar.success("✅ 校准系统初始化成功")
    else:
        st.sidebar.warning("⚠️ 校准系统初始化可能存在问题")
    return calibrator, success

# 初始化校准系统
calibrator, calibration_success = initialize_calibration()

# 标题
st.title("太极拳动作识别与评分系统")

# 侧边栏 - 选择标准动作
st.sidebar.header("选择标准动作")

# 评分校准设置
st.sidebar.subheader("评分校准设置")
enable_calibration = st.sidebar.checkbox("启用评分校准", value=True, help="启用后将自动调整评分至适合教学的水平")
if enable_calibration:
    st.sidebar.info("📊 校准已启用：评分将调整至教学友好水平")
else:
    st.sidebar.warning("⚠️ 校准已禁用：使用原始严格评分")

standard_folder = os.path.join(folders["processed_dir"], "standard")
standard_video_folder = folders["standard_dir"]

# 检查标准动作文件夹是否存在
if not os.path.exists(standard_folder):
    st.sidebar.error(f"标准动作处理文件夹不存在: {standard_folder}")
    st.sidebar.info("请先运行 data_process.py 处理标准动作视频")
    standard_files = []
else:
    # 获取所有.npy文件（关键点数据）
    standard_files = [f for f in os.listdir(standard_folder) if f.endswith("_landmarks.npy")]

# 检查标准视频文件夹
standard_videos = []
if os.path.exists(standard_video_folder):
    standard_videos = [f for f in os.listdir(standard_video_folder) 
                      if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

# 选择标准动作方式
standard_selection_method = st.sidebar.radio(
    "选择标准动作方式",
    ["使用已处理的标准动作", "上传新的标准动作视频"],
    index=0 if standard_files else 1
)

standard_landmarks = None
selected_standard_path = None

if standard_selection_method == "使用已处理的标准动作":
    if standard_files:
        selected_standard = st.sidebar.selectbox("选择标准动作", standard_files)
        standard_path = os.path.join(standard_folder, selected_standard)
        st.sidebar.success(f"已加载标准动作: {selected_standard}")
        
        # 加载标准动作数据
        standard_landmarks = np.load(standard_path, allow_pickle=True)
        st.sidebar.info(f"标准动作帧数: {len(standard_landmarks)}")
    else:
        st.sidebar.warning("未找到已处理的标准动作数据，请上传新的标准动作视频")
else:
    if standard_videos:
        selected_standard_video = st.sidebar.selectbox("选择标准动作视频", standard_videos)
        selected_standard_path = os.path.join(standard_video_folder, selected_standard_video)
        st.sidebar.success(f"已选择标准动作视频: {selected_standard_video}")
    else:
        uploaded_standard = st.sidebar.file_uploader("上传标准动作视频", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_standard is not None:
            # 保存上传的视频到标准动作文件夹
            standard_video_path = os.path.join(standard_video_folder, uploaded_standard.name)
            with open(standard_video_path, "wb") as f:
                f.write(uploaded_standard.getbuffer())
            selected_standard_path = standard_video_path
            st.sidebar.success(f"已上传标准动作视频: {uploaded_standard.name}")
    
    # 处理标准动作视频
    if selected_standard_path and st.sidebar.button("处理标准动作视频", key="process_standard_button"):
        with st.spinner("正在处理标准动作视频..."):
            # 提取标准动作关键点
            output_folder = os.path.join(folders["processed_dir"], "standard")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # --- 🔥 修改开始：添加标准动作的 FPS 动态计算 ---
            # 1. 计算视频时长
            cap_temp = cv2.VideoCapture(selected_standard_path)
            fps_temp = cap_temp.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps_temp if fps_temp > 0 else 0
            cap_temp.release()
            
            # 2. 设置目标采样率
            # 如果视频超过60秒，强制使用 3.0 FPS；否则使用 10.0 FPS
            current_standard_fps = 3.0 if duration > 60 else 10.0
            
            st.sidebar.info(f"视频时长 {duration:.1f}s，采样率: {current_standard_fps} FPS")
            
            # 3. 传递 target_fps 参数 (您之前的代码漏了这一步)
            standard_landmarks = extract_landmarks(
                selected_standard_path, 
                output_folder, 
                target_fps=current_standard_fps  # <--- 必须加上这个参数
            )
            # --- 🔥 修改结束 ---
            
            st.sidebar.success(f"标准动作处理完成，提取了{len(standard_landmarks)}帧的关键点数据")
# 主界面 - 上传或录制学生动作视频
st.header("上传或录制学生动作视频")

tab1, tab2 = st.tabs(["上传视频", "实时录制"])

with tab1:
    uploaded_file = st.file_uploader("上传学生动作视频", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # 保存上传的视频到临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        temp_file.close()
        
        # 显示上传的视频
        st.video(video_path)
        
# 处理按钮
        if st.button("开始评分", key="score_button") and standard_landmarks is not None:
            with st.spinner("正在处理视频..."):
                # 提取学生动作关键点
                output_folder = os.path.join(folders["processed_dir"], "student")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                # --- 新增：动态计算 FPS ---
                cap_temp = cv2.VideoCapture(video_path)
                fps_temp = cap_temp.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps_temp if fps_temp > 0 else 0
                cap_temp.release()
                
                # 同样的逻辑：长视频降频处理
                current_fps = 3.0
                st.info(f"检测到视频时长 {duration:.1f}s，自动调整采样率为: {current_fps} FPS")
                # ------------------------
                
                # 这里将原来的 10.0 修改为 current_fps
                student_landmarks = extract_landmarks(video_path, output_folder, target_fps=current_fps)
                # 使用带校准功能的评分函数
                result = score_movement_improved_with_calibration(
                    student_landmarks, 
                    standard_landmarks, 
                    enable_calibration=enable_calibration
                )
                score = result['total_score']
                errors = result['errors']
                paths = result.get('paths', [])
                
                # 获取改进建议
                suggestions = get_error_suggestions_improved(errors)
                
                # 可视化比较
                results_folder = folders["results_dir"]
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder)
                
                comparison_path = os.path.join(results_folder, "comparison.png")
                # 从paths字典中选择position特征的路径用于可视化
                visualization_path = paths.get('position', []) if isinstance(paths, dict) else paths
                
                # 使用新的姿态可视化功能
                try:
                    pose_comparison_images = create_pose_comparison_visualization(
                        video_path, student_landmarks, standard_landmarks, results_folder
                    )
                    visualization_success = True
                except Exception as e:
                    print(f"新可视化功能失败，使用原有功能: {e}")
                    # 回退到原有的可视化功能
                    if len(visualization_path) > 0:  # 只有当路径列表不为空时才进行可视化
                        visualize_comparison(student_landmarks, standard_landmarks, visualization_path, comparison_path)
                    visualization_success = len(visualization_path) > 0 and os.path.exists(comparison_path)
                
                # 显示评分结果
                if result.get('calibration_applied', False):
                    st.success(f"评分结果: {score:.1f}/100 (已校准)")
                    if 'original_score' in result:
                        st.info(f"原始评分: {result['original_score']:.1f}/100")
                else:
                    st.success(f"评分结果: {score:.1f}/100 (原始评分)")
                
                # 显示错误分析
                st.subheader("动作错误分析")
                for error in errors:
                    severity = error["severity"]
                    if severity == "严重偏差":
                        st.error(f"{error['feature']} - {severity}")
                    elif severity == "中度偏差":
                        st.warning(f"{error['feature']} - {severity}")
                    else:
                        st.info(f"{error['feature']} - {severity}")
                
                # 显示改进建议
                st.subheader("改进建议")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
                
                # 显示比较图（只有在可视化成功生成时才显示）
                if visualization_success:
                    if 'pose_comparison_images' in locals() and pose_comparison_images:
                        st.subheader("动作比较可视化")
                        st.write("红色：学生动作，绿色：标准动作")
                        
                        # 显示可视化图片（增加到6张）
                        cols = st.columns(min(3, len(pose_comparison_images)))
                        for i, img_path in enumerate(pose_comparison_images[:6]):  # 限制最多显示6张
                            with cols[i % len(cols)]:
                                st.image(img_path, caption=f"关键帧 {i+1}")
                    elif os.path.exists(comparison_path):
                        st.image(comparison_path, caption="动作比较（上：学生，下：标准）")
                else:
                    st.info("动作比较图生成失败，但评分结果仍然有效")
        elif standard_landmarks is None and st.button("开始评分", key="score_button_error"):
            st.error("请先选择或处理标准动作视频")

with tab2:
    st.write("实时录制功能")
    
    # 检测是否在服务器环境
    try:
        import platform
        is_server = platform.system() == "Linux" and not os.path.exists("/dev/video0")
    except:
        is_server = True
    
    if is_server:
        st.warning("⚠️ 检测到服务器环境，实时录制功能不可用")
        st.info("💡 建议使用\"上传视频\"功能，或在本地环境运行系统以使用实时录制")
        st.markdown("""
        **在本地环境使用实时录制功能：**
        1. 在本地电脑上运行 `streamlit run app.py`
        2. 确保摄像头已连接并可用
        3. 点击"开始录制"按钮进行实时录制
        """)
    else:
        st.write("准备好后点击开始录制")
    
        if st.button("开始录制", key="record_button") and standard_landmarks is not None:
            # 创建输出文件夹
            output_folder = os.path.join(folders["processed_dir"], "student")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # 尝试打开摄像头
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ 无法访问摄像头，请检查：")
                st.markdown("""
                - 摄像头是否已连接
                - 摄像头是否被其他应用占用
                - 浏览器是否有摄像头权限
                """)
                cap.release()
            else:
                # 设置视频保存
                timestamp = int(time.time())
                video_path = os.path.join(folders["student_dir"], f"student_{timestamp}.mp4")
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = 30
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                # 创建占位符
                video_placeholder = st.empty()
                
                # 录制状态
                recording = True
                start_time = time.time()
                max_duration = 30  # 最长录制30秒
                
                student_landmarks = []
                
                # 录制循环
                try:
                    while recording and (time.time() - start_time) < max_duration:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 处理帧
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)
                        
                        if results.pose_landmarks:
                            # 绘制关键点
                            mp_drawing.draw_landmarks(
                                frame, 
                                results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS
                            )
                            
                            # 提取关键点数据
                            frame_landmarks = []
                            for landmark in results.pose_landmarks.landmark:
                                frame_landmarks.append({
                                    'x': landmark.x,
                                    'y': landmark.y,
                                    'z': landmark.z,
                                    'visibility': landmark.visibility
                                })
                            student_landmarks.append(frame_landmarks)
                        
                        # 显示剩余时间
                        remaining = max_duration - (time.time() - start_time)
                        cv2.putText(frame, f"剩余时间: {remaining:.1f}秒", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # 保存帧
                        out.write(frame)
                        
                        # 显示帧（注意：在服务器环境可能不会实时更新）
                        video_placeholder.image(frame, channels="BGR")
                
                finally:
                    # 释放资源
                    cap.release()
                    out.release()
                    
                    # 保存关键点数据
                    if student_landmarks:
                        landmarks_path = os.path.join(output_folder, f"student_{timestamp}_landmarks.npy")
                        np.save(landmarks_path, student_landmarks)
                        
                        # 使用带校准功能的评分函数
                        result = score_movement_improved_with_calibration(
                            student_landmarks, 
                            standard_landmarks, 
                            enable_calibration=enable_calibration
                        )
                        score = result['total_score']
                        errors = result['errors']
                        path = result.get('paths', [])
                        
                        # 获取改进建议
                        suggestions = get_error_suggestions_improved(errors)
                        
                        # 可视化比较
                        results_folder = folders["results_dir"]
                        if not os.path.exists(results_folder):
                            os.makedirs(results_folder)
                        
                        comparison_path = os.path.join(results_folder, f"comparison_{timestamp}.png")
                        
                        # 使用新的姿态可视化功能
                        try:
                            pose_comparison_images = create_pose_comparison_visualization(
                                video_path, student_landmarks, standard_landmarks, results_folder
                            )
                            visualization_success = True
                        except Exception as e:
                            print(f"新可视化功能失败，使用原有功能: {e}")
                            # 回退到原有的可视化功能
                            visualize_comparison(student_landmarks, standard_landmarks, path, comparison_path)
                            visualization_success = os.path.exists(comparison_path)
                        
                        # 显示评分结果
                        if result.get('calibration_applied', False):
                            st.success(f"评分结果: {score:.1f}/100 (已校准)")
                            if 'original_score' in result:
                                st.info(f"原始评分: {result['original_score']:.1f}/100")
                        else:
                            st.success(f"评分结果: {score:.1f}/100 (原始评分)")
                        
                        # 显示错误分析
                        st.subheader("动作错误分析")
                        for error in errors:
                            severity = error["severity"]
                            if severity == "严重偏差":
                                st.error(f"{error['feature']} - {severity}")
                            elif severity == "中度偏差":
                                st.warning(f"{error['feature']} - {severity}")
                            else:
                                st.info(f"{error['feature']} - {severity}")
                        
                        # 显示改进建议
                        st.subheader("改进建议")
                        for suggestion in suggestions:
                            st.write(f"- {suggestion}")
                        
                        # 显示比较图（只有在可视化成功生成时才显示）
                        if visualization_success:
                            if 'pose_comparison_images' in locals() and pose_comparison_images:
                                st.subheader("动作比较可视化")
                                st.write("红色：学生动作，绿色：标准动作")
                                
                                # 显示可视化图片（增加到6张）
                                cols = st.columns(min(3, len(pose_comparison_images)))
                                for i, img_path in enumerate(pose_comparison_images[:6]):  # 限制最多显示6张
                                    with cols[i % len(cols)]:
                                        st.image(img_path, caption=f"关键帧 {i+1}")
                            elif os.path.exists(comparison_path):
                                st.image(comparison_path, caption="动作比较（上：学生，下：标准）")
                        else:
                            st.info("动作比较图生成失败，但评分结果仍然有效")
                        
                        # 显示录制的视频
                        st.video(video_path)
                    else:
                        st.error("未检测到有效的姿势数据，请重试")
        elif standard_landmarks is None and st.button("开始录制", key="record_button_error"):
            st.error("请先选择或处理标准动作视频")

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 使用步骤：
    1. 在左侧边栏选择标准动作或上传新的标准动作视频
    2. 如果上传了新的标准动作视频，点击"处理标准动作视频"按钮
    3. 选择上传视频或实时录制
    4. 上传视频：上传一段太极拳动作视频，然后点击"开始评分"
    5. 实时录制：点击"开始录制"，站在摄像头前做太极拳动作，系统会自动录制30秒
    6. 系统会自动评分并提供错误分析和改进建议
    
    ### 注意事项：
    - 请确保光线充足，背景简单
    - 尽量穿着与背景对比明显的衣服
    - 确保全身在摄像头视野内
    - 动作速度应与标准动作相近
    - 录制时保持稳定，避免快速移动
    """)

# 添加数据处理教程
with st.expander("数据处理详细教程"):
    st.markdown("""
    ## 太极拳动作数据处理详细教程
    
    ### 1. 准备工作
    
    #### 1.1 安装必要的软件和库
    
    首先，确保您已安装以下软件：
    - Python 3.8或更高版本
    - pip（Python包管理器）
    
    然后，安装必要的Python库：
    ```
    pip install mediapipe opencv-python numpy matplotlib fastdtw scipy streamlit
    ```
    
    #### 1.2 准备视频数据
    
    准备两类视频：
    - **标准动作视频**：由太极拳大师或教师演示的标准动作
    - **学生动作视频**：需要评分的学生动作
    
    视频要求：
    - 分辨率不低于720p
    - 光线充足，背景简单
    - 演示者穿着与背景对比明显的衣服
    - 确保全身在画面内
    - 每个动作单独录制，长度10-30秒
    
    ### 2. 数据处理流程
    
    #### 2.1 提取关键点
    
    系统使用MediaPipe Pose模型从视频中提取人体关键点。每个关键点包含以下信息：
    - x, y, z坐标（归一化到0-1范围）
    - 可见性得分（0-1，表示关键点的可信度）
    
    MediaPipe可以识别33个人体关键点，包括：
    - 面部关键点（0-10）
    - 上肢关键点（11-22）：肩膀、肘部、手腕等
    - 躯干关键点（23-24）：髋部
    - 下肢关键点（25-32）：膝盖、脚踝等
    
    #### 2.2 数据标准化
    
    为了消除身高、距离等因素的影响，系统会对提取的关键点进行标准化：
    - 以髋部中心为参考点
    - 根据身体比例（髋部到肩部的距离）进行缩放
    - 这样可以确保不同体型的人动作可以进行公平比较
    
    #### 2.3 特征提取
    
    系统从标准化的关键点数据中提取以下特征：
    - **关节位置**：关键关节（肩、肘、腕、髋、膝、踝）的3D坐标
    - **关节角度**：计算重要关节（肘部、膝盖、躯干）的角度
    
    这些特征构成了评分的基础。
    
    ### 3. 评分系统
    
    #### 3.1 动态时间规整（DTW）
    
    系统使用FastDTW算法（DTW的改进版）比较学生动作和标准动作：
    - DTW可以处理动作速度不一致的情况
    - 它找到两个时间序列之间的最佳对应关系
    - 计算对应点之间的加权距离
    
    #### 3.2 评分计算
    
    评分基于DTW距离计算：
    - 距离越小，评分越高
    - 评分范围：0-100分
    - 不同特征有不同的权重（位置特征0.7，角度特征0.3）
    
    #### 3.3 错误分析
    
    系统会分析学生动作中的主要错误：
    - 识别误差最大的关节或角度
    - 根据误差大小分为轻微、中度和严重偏差
    - 提供针对性的改进建议
    
    ### 4. 使用自定义数据
    
    #### 4.1 添加新的标准动作
    
    1. 将标准动作视频放入 `data/standard` 文件夹
    2. 运行 `data_process.py` 脚本处理视频
    3. 系统会自动提取关键点并保存到 `data/processed/standard` 文件夹
    
    #### 4.2 处理学生动作
    
    1. 将学生动作视频放入 `data/student` 文件夹
    2. 运行 `data_process.py` 脚本处理视频
    3. 系统会自动提取关键点并保存到 `data/processed/student` 文件夹
    
    #### 4.3 评分和分析
    
    1. 运行 `scoring.py` 脚本，指定学生动作和标准动作文件
    2. 系统会生成评分报告、比较可视化和改进建议
    3. 结果保存在 `results` 文件夹中
    
    ### 5. 使用Web界面
    
    1. 运行 `streamlit run app.py` 启动Web界面
    2. 在浏览器中访问 http://localhost:8501
    3. 按照界面提示上传视频或实时录制动作
    4. 查看评分结果和改进建议
    
    ### 6. 高级自定义
    
    #### 6.1 调整评分权重
    
    您可以在 `scoring.py` 文件中修改特征权重：
    ```python
    # 默认权重：位置特征权重0.7，角度特征权重0.3
    feature_count = student_features.shape[1]
    position_count = feature_count - 5  # 减去5个角度特征
    weights = np.ones(feature_count)
    weights[:position_count] *= 0.7 / position_count
    weights[position_count:] *= 0.3 / 5
    ```
    
    #### 6.2 添加新的特征
    
    您可以在 `scoring.py` 文件的 `calculate_joint_angles` 函数中添加新的角度计算：
    ```python
    # 计算新的关节角度
    # ...
    frame_angles['new_angle'] = new_angle
    ```
    
    然后在 `feature_names` 列表中添加新特征的名称，并在 `get_error_suggestions` 函数中添加相应的改进建议。
    
    #### 6.3 调整可视化设置
    
    您可以在 `data_process.py` 和 `scoring.py` 文件中修改可视化函数的参数，如图像大小、颜色、标题等。
    """)

# 添加项目信息
st.sidebar.markdown("---")
st.sidebar.info("""
### 太极拳动作识别与评分系统

本系统使用计算机视觉技术识别太极拳动作，并与标准动作进行比较评分。

技术栈：
- MediaPipe (姿态估计)
- FastDTW (动作比较)
- Streamlit (用户界面)

版本：1.0.0
""")