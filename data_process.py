import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt

# 初始化MediaPipe姿势检测
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

def cv2_imwrite_chinese(file_path, img):
    """
    解决OpenCV无法保存带有中文路径图片的问题
    """
    try:
        # 使用imencode将图片编码，然后用tofile保存，这样可以支持中文路径
        cv2.imencode('.jpg', img)[1].tofile(file_path)
        return True
    except Exception as e:
        print(f"保存图片失败: {e}")
        return False

def interpolate_missing_landmarks(landmarks_data, visibility_threshold=0.5):
    """
    对缺失或低可见度的关节点进行插值处理
    
    参数:
    landmarks_data: 关键点数据列表
    visibility_threshold: 可见性阈值，低于此值的关键点将被插值
    
    返回:
    interpolated_data: 插值后的关键点数据
    """
    # 修复NumPy数组判断问题
    if landmarks_data is None or (isinstance(landmarks_data, list) and len(landmarks_data) == 0):
        return landmarks_data
    
    interpolated_data = [frame.copy() for frame in landmarks_data]
    num_frames = len(landmarks_data)
    num_landmarks = len(landmarks_data[0])
    
    print(f"开始插值处理，共{num_frames}帧，{num_landmarks}个关键点")
    
    # 对每个关节点进行处理
    for landmark_idx in range(num_landmarks):
        # 找出该关节点在所有帧中的可见性
        visibility = [frame[landmark_idx]['visibility'] for frame in landmarks_data]
        
        # 找出需要插值的帧（可见性低于阈值）
        frames_to_interpolate = [i for i, v in enumerate(visibility) if v < visibility_threshold]
        
        if not frames_to_interpolate:
            continue
            
        interpolated_count = 0
        # 对每个需要插值的帧
        for frame_idx in frames_to_interpolate:
            # 寻找前后有效的帧
            prev_valid_idx = None
            next_valid_idx = None
            
            # 向前找
            for i in range(frame_idx-1, -1, -1):
                if visibility[i] >= visibility_threshold:
                    prev_valid_idx = i
                    break
                    
            # 向后找
            for i in range(frame_idx+1, num_frames):
                if visibility[i] >= visibility_threshold:
                    next_valid_idx = i
                    break
            
            # 如果前后都有有效帧，进行线性插值
            if prev_valid_idx is not None and next_valid_idx is not None:
                weight = (frame_idx - prev_valid_idx) / (next_valid_idx - prev_valid_idx)
                
                for coord in ['x', 'y', 'z']:
                    prev_value = landmarks_data[prev_valid_idx][landmark_idx][coord]
                    next_value = landmarks_data[next_valid_idx][landmark_idx][coord]
                    interpolated_value = prev_value + weight * (next_value - prev_value)
                    interpolated_data[frame_idx][landmark_idx][coord] = interpolated_value
                
                # 设置插值后的可见性值
                interpolated_data[frame_idx][landmark_idx]['visibility'] = 0.8
                interpolated_count += 1
            
            # 如果只有前面有效帧，使用前面的值
            elif prev_valid_idx is not None:
                for coord in ['x', 'y', 'z']:
                    interpolated_data[frame_idx][landmark_idx][coord] = landmarks_data[prev_valid_idx][landmark_idx][coord]
                interpolated_data[frame_idx][landmark_idx]['visibility'] = 0.7
                interpolated_count += 1
            
            # 如果只有后面有效帧，使用后面的值
            elif next_valid_idx is not None:
                for coord in ['x', 'y', 'z']:
                    interpolated_data[frame_idx][landmark_idx][coord] = landmarks_data[next_valid_idx][landmark_idx][coord]
                interpolated_data[frame_idx][landmark_idx]['visibility'] = 0.7
                interpolated_count += 1
        
        if interpolated_count > 0:
            print(f"关键点{landmark_idx}插值了{interpolated_count}帧")
    
    return interpolated_data

def smooth_landmarks(landmarks_data, window_size=5):
    """
    对关键点数据进行平滑处理
    
    参数:
    landmarks_data: 关键点数据列表
    window_size: 平滑窗口大小
    
    返回:
    smoothed_data: 平滑后的关键点数据
    """
    if landmarks_data is None or len(landmarks_data) < window_size:
        return landmarks_data
    
    smoothed_data = [frame.copy() for frame in landmarks_data]
    num_frames = len(landmarks_data)
    num_landmarks = len(landmarks_data[0])
    
    print(f"开始平滑处理，窗口大小：{window_size}")
    
    # 对每个关键点进行平滑
    for landmark_idx in range(num_landmarks):
        for coord in ['x', 'y', 'z']:
            # 提取该关键点在所有帧中的坐标值
            coord_values = [frame[landmark_idx][coord] for frame in landmarks_data]
            
            # 应用移动平均平滑
            for frame_idx in range(num_frames):
                start_idx = max(0, frame_idx - window_size // 2)
                end_idx = min(num_frames, frame_idx + window_size // 2 + 1)
                
                # 计算窗口内的平均值
                window_values = coord_values[start_idx:end_idx]
                smoothed_value = sum(window_values) / len(window_values)
                
                smoothed_data[frame_idx][landmark_idx][coord] = smoothed_value
    
    return smoothed_data

def extract_landmarks(video_path, output_folder, visualize=True, target_fps=10.0):
    """
    从视频中提取姿势关键点并保存，使用统一的时间采样策略
    
    参数:
    video_path: 视频文件路径
    output_folder: 输出文件夹路径
    visualize: 是否可视化关键点并保存图像
    target_fps: 目标采样频率（帧/秒），默认10 FPS确保时间一致性
    
    返回:
    landmarks_data: 关键点数据列表
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    landmarks_data = []
    
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建可视化文件夹
    vis_folder = os.path.join(output_folder, f"{video_name}_vis")
    if visualize and not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
    
    # 获取视频参数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算采样间隔（基于时间而非固定帧数）
    if original_fps > 0:
        frame_interval = max(1, int(original_fps / target_fps))
    else:
        frame_interval = 3  # 默认值，兼容原有逻辑
    
    print(f"视频总帧数: {total_frames}")
    print(f"原始FPS: {original_fps:.2f}")
    print(f"目标采样FPS: {target_fps}")
    print(f"计算得出的帧间隔: {frame_interval}")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # 基于时间的统一采样策略
        if frame_count % frame_interval == 0:
            print(f"处理帧 {frame_count}/{total_frames}")
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 处理图像
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # 绘制关键点
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
                # 保存带关键点的图像
                if visualize:
                    output_path = os.path.join(vis_folder, f"frame_{frame_count:04d}.jpg")
                    # 使用支持中文路径的函数替代
                    cv2_imwrite_chinese(output_path, annotated_image)
                    
                # 提取关键点数据
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                landmarks_data.append(frame_landmarks)
        
        frame_count += 1
        
        # 每处理100帧显示进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    cap.release()
    
    # 对提取的关键点数据进行插值和平滑处理
    if landmarks_data:
        print("\n开始关键点数据后处理...")
        
        # 1. 插值处理缺失的关键点
        landmarks_data = interpolate_missing_landmarks(landmarks_data, visibility_threshold=0.6)
        
        # 2. 平滑处理关键点数据
        landmarks_data = smooth_landmarks(landmarks_data, window_size=5)
        
        print("关键点数据后处理完成\n")
    
    # 保存关键点数据
    landmarks_path = os.path.join(output_folder, f"{video_name}_landmarks.npy")
    np.save(landmarks_path, landmarks_data)
    print(f"处理完成：{video_path}，提取了{len(landmarks_data)}帧的关键点数据")
    print(f"关键点数据已保存至：{landmarks_path}")
    
    return landmarks_data

def visualize_landmarks(landmarks_data, output_folder, video_name, num_frames=5):
    """
    可视化关键点数据
    
    参数:
    landmarks_data: 关键点数据列表
    output_folder: 输出文件夹路径
    video_name: 视频名称
    num_frames: 要可视化的帧数
    """
    if landmarks_data is None:
        print("没有关键点数据可供可视化")
        return
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 选择要可视化的帧
    total_frames = len(landmarks_data)
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    plt.figure(figsize=(15, 3*num_frames))
    for i, idx in enumerate(frame_indices):
        plt.subplot(num_frames, 1, i+1)
        plt.title(f"帧 {idx}")
        
        # 绘制关键点
        frame_landmarks = landmarks_data[idx]
        x_coords = []
        y_coords = []
        visibility = []
        
        for landmark in frame_landmarks:
            x_coords.append(landmark['x'])
            y_coords.append(landmark['y'])
            visibility.append(landmark['visibility'])
        
        # 绘制身体连接线
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx < len(frame_landmarks) and end_idx < len(frame_landmarks):
                if frame_landmarks[start_idx]['visibility'] > 0.5 and frame_landmarks[end_idx]['visibility'] > 0.5:
                    plt.plot(
                        [x_coords[start_idx], x_coords[end_idx]],
                        [y_coords[start_idx], y_coords[end_idx]],
                        'g-'
                    )
        
        # 绘制关键点
        for j in range(len(frame_landmarks)):
            if visibility[j] > 0.5:  # 只绘制可见性高的点
                plt.plot(x_coords[j], y_coords[j], 'ro')
        
        plt.xlim(0, 1)
        plt.ylim(1, 0)  # 反转y轴，使图像方向正确
        plt.axis('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{video_name}_landmarks_vis.png")
    plt.savefig(output_path)
    plt.close()
    print(f"关键点可视化已保存至：{output_path}")

def analyze_landmarks(landmarks_data):
    """
    分析关键点数据，提取有用的统计信息
    
    参数:
    landmarks_data: 关键点数据列表
    
    返回:
    stats: 统计信息字典
    """
    if landmarks_data is None:
        return {}
    
    stats = {}
    
    # 计算关节角度随时间的变化
    right_elbow_angles = []
    left_elbow_angles = []
    right_knee_angles = []
    left_knee_angles = []
    
    for frame in landmarks_data:
        # 计算右肘角度
        right_shoulder = np.array([frame[12]['x'], frame[12]['y'], frame[12]['z']])
        right_elbow = np.array([frame[14]['x'], frame[14]['y'], frame[14]['z']])
        right_wrist = np.array([frame[16]['x'], frame[16]['y'], frame[16]['z']])
        
        v1 = right_shoulder - right_elbow
        v2 = right_wrist - right_elbow
        right_elbow_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        right_elbow_angles.append(right_elbow_angle)
        
        # 计算左肘角度
        left_shoulder = np.array([frame[11]['x'], frame[11]['y'], frame[11]['z']])
        left_elbow = np.array([frame[13]['x'], frame[13]['y'], frame[13]['z']])
        left_wrist = np.array([frame[15]['x'], frame[15]['y'], frame[15]['z']])
        
        v1 = left_shoulder - left_elbow
        v2 = left_wrist - left_elbow
        left_elbow_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        left_elbow_angles.append(left_elbow_angle)
        
        # 计算右膝角度
        right_hip = np.array([frame[24]['x'], frame[24]['y'], frame[24]['z']])
        right_knee = np.array([frame[26]['x'], frame[26]['y'], frame[26]['z']])
        right_ankle = np.array([frame[28]['x'], frame[28]['y'], frame[28]['z']])
        
        v1 = right_hip - right_knee
        v2 = right_ankle - right_knee
        right_knee_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        right_knee_angles.append(right_knee_angle)
        
        # 计算左膝角度
        left_hip = np.array([frame[23]['x'], frame[23]['y'], frame[23]['z']])
        left_knee = np.array([frame[25]['x'], frame[25]['y'], frame[25]['z']])
        left_ankle = np.array([frame[27]['x'], frame[27]['y'], frame[27]['z']])
        
        v1 = left_hip - left_knee
        v2 = left_ankle - left_knee
        left_knee_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        left_knee_angles.append(left_knee_angle)
    
    # 保存角度数据
    stats['right_elbow_angles'] = right_elbow_angles
    stats['left_elbow_angles'] = left_elbow_angles
    stats['right_knee_angles'] = right_knee_angles
    stats['left_knee_angles'] = left_knee_angles
    
    # 计算统计信息
    stats['avg_right_elbow_angle'] = np.mean(right_elbow_angles)
    stats['avg_left_elbow_angle'] = np.mean(left_elbow_angles)
    stats['avg_right_knee_angle'] = np.mean(right_knee_angles)
    stats['avg_left_knee_angle'] = np.mean(left_knee_angles)
    
    return stats

def visualize_angles(stats, output_folder, video_name):
    """
    可视化关节角度随时间的变化
    
    参数:
    stats: 统计信息字典
    output_folder: 输出文件夹路径
    video_name: 视频名称
    """
    if not stats:
        print("没有统计数据可供可视化")
        return
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.figure(figsize=(15, 10))
    
    # 绘制右肘角度
    plt.subplot(2, 2, 1)
    plt.plot(stats['right_elbow_angles'])
    plt.title('右肘角度')
    plt.xlabel('帧')
    plt.ylabel('角度 (度)')
    plt.grid(True)
    
    # 绘制左肘角度
    plt.subplot(2, 2, 2)
    plt.plot(stats['left_elbow_angles'])
    plt.title('左肘角度')
    plt.xlabel('帧')
    plt.ylabel('角度 (度)')
    plt.grid(True)
    
    # 绘制右膝角度
    plt.subplot(2, 2, 3)
    plt.plot(stats['right_knee_angles'])
    plt.title('右膝角度')
    plt.xlabel('帧')
    plt.ylabel('角度 (度)')
    plt.grid(True)
    
    # 绘制左膝角度
    plt.subplot(2, 2, 4)
    plt.plot(stats['left_knee_angles'])
    plt.title('左膝角度')
    plt.xlabel('帧')
    plt.ylabel('角度 (度)')
    plt.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{video_name}_angles_vis.png")
    plt.savefig(output_path)
    plt.close()
    print(f"关节角度可视化已保存至：{output_path}")

# 主函数
def process_video(video_path, output_base_folder="data"):
    """
    处理单个视频文件，提取关键点并进行可视化分析
    
    参数:
    video_path: 视频文件路径
    output_base_folder: 输出基础文件夹路径
    """
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建输出文件夹
    output_folder = os.path.join(output_base_folder, video_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"开始处理视频: {video_path}")
    print(f"输出文件夹: {output_folder}")
    
    # 提取关键点
    landmarks_data = extract_landmarks(video_path, output_folder)
    
    # 可视化关键点
    visualize_landmarks(landmarks_data, output_folder, video_name)
    
    # 分析关键点
    stats = analyze_landmarks(landmarks_data)
    
    # 可视化关节角度
    visualize_angles(stats, output_folder, video_name)
    
    print(f"视频处理完成: {video_path}")
    return landmarks_data, stats

def process_folder(folder_path, output_base_folder="data"):
    """
    处理文件夹中的所有视频文件
    
    参数:
    folder_path: 视频文件夹路径
    output_base_folder: 输出基础文件夹路径
    """
    # 创建输出基础文件夹
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    # 获取所有视频文件
    video_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_files.append(os.path.join(folder_path, filename))
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    results = {}
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        landmarks_data, stats = process_video(video_path, output_base_folder)
        results[video_name] = {
            'landmarks_data': landmarks_data,
            'stats': stats
        }
    
    return results

# 使用示例
if __name__ == "__main__":
    # 创建数据文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 创建标准动作和学生动作文件夹
    standard_dir = os.path.join(data_dir, "standard")
    student_dir = os.path.join(data_dir, "student")
    
    if not os.path.exists(standard_dir):
        os.makedirs(standard_dir)
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)
    
    print("太极拳动作数据处理工具")
    print("====================")
    print("1. 请将标准动作视频放在 data/standard 文件夹中")
    print("2. 请将学生动作视频放在 data/student 文件夹中")
    print("3. 运行此脚本处理视频")
    print()
    
    choice = input("请选择要处理的视频类型 (1: 标准动作, 2: 学生动作, 3: 全部处理): ")
    
    if choice == "1":
        print("\n处理标准动作视频...")
        process_folder(standard_dir, os.path.join(data_dir, "processed", "standard"))
    elif choice == "2":
        print("\n处理学生动作视频...")
        process_folder(student_dir, os.path.join(data_dir, "processed", "student"))
    elif choice == "3":
        print("\n处理标准动作视频...")
        process_folder(standard_dir, os.path.join(data_dir, "processed", "standard"))
        print("\n处理学生动作视频...")
        process_folder(student_dir, os.path.join(data_dir, "processed", "student"))
    else:
        print("无效的选择")