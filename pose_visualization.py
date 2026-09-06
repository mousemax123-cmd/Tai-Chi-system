import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import os
from typing import List, Dict, Tuple, Optional
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class PoseVisualizer:
    """改进的姿态可视化器，确保骨骼点与视频背景准确对应"""
    
    def __init__(self):
        # MediaPipe姿态连接定义
        self.pose_connections = [
            # 面部
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # 躯干
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 左臂
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # 右臂
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # 左腿
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # 右腿
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        
        # 关键关节点索引（用于差异计算）
        self.key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    def extract_frame_landmarks_from_video(self, video_path: str, frame_indices: List[int]) -> List[List[Dict]]:
        """
        从视频中提取指定帧的关键点数据
        
        参数:
        video_path: 视频文件路径
        frame_indices: 要提取的帧索引列表
        
        返回:
        frame_landmarks: 每帧的关键点数据列表
        """
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frame_landmarks = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"警告: 无法读取第{frame_idx}帧")
                frame_landmarks.append([])
                continue
            
            # 处理帧以提取关键点
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                frame_landmarks.append(landmarks)
            else:
                frame_landmarks.append([])
        
        cap.release()
        pose.close()
        return frame_landmarks
    
    def calculate_pose_difference(self, student_landmarks: List[Dict], 
                                standard_landmarks: List[Dict]) -> List[float]:
        """
        计算学生动作与标准动作的差异
        """
        min_length = min(len(student_landmarks), len(standard_landmarks))
        differences = []
        
        for i in range(min_length):
            student_frame = student_landmarks[i]
            standard_frame = standard_landmarks[i]
            
            if not student_frame or not standard_frame:
                differences.append(float('inf'))
                continue
            
            frame_diff = 0.0
            valid_joints = 0
            
            # 计算关键关节点的位置差异
            for joint_idx in self.key_joints:
                if (joint_idx < len(student_frame) and 
                    joint_idx < len(standard_frame) and
                    student_frame[joint_idx]['visibility'] > 0.5 and
                    standard_frame[joint_idx]['visibility'] > 0.5):
                    
                    student_pos = np.array([
                        student_frame[joint_idx]['x'],
                        student_frame[joint_idx]['y'],
                        student_frame[joint_idx]['z']
                    ])
                    
                    standard_pos = np.array([
                        standard_frame[joint_idx]['x'],
                        standard_frame[joint_idx]['y'],
                        standard_frame[joint_idx]['z']
                    ])
                    
                    # 计算欧氏距离
                    diff = np.linalg.norm(student_pos - standard_pos)
                    frame_diff += diff
                    valid_joints += 1
            
            # 计算平均差异
            if valid_joints > 0:
                frame_diff /= valid_joints
            else:
                frame_diff = float('inf')
            
            differences.append(frame_diff)
        
        return differences
    
    def select_key_frames_with_dtw(self, student_landmarks: List[Dict], 
                                  standard_landmarks: List[Dict], 
                                  num_frames: int = 6) -> List[Tuple[int, int]]:
        """
        使用DTW算法选择关键帧对，确保时间对应关系
        
        返回:
        key_frame_pairs: [(student_frame_idx, standard_frame_idx), ...]
        """
        # 提取关键特征用于DTW
        def extract_features(landmarks_list):
            features = []
            for frame_landmarks in landmarks_list:
                if frame_landmarks is None or len(frame_landmarks) == 0:
                    features.append(np.zeros(len(self.key_joints) * 3))
                    continue
                
                frame_features = []
                for joint_idx in self.key_joints:
                    if (joint_idx < len(frame_landmarks) and 
                        frame_landmarks[joint_idx]['visibility'] > 0.5):
                        frame_features.extend([
                            frame_landmarks[joint_idx]['x'],
                            frame_landmarks[joint_idx]['y'],
                            frame_landmarks[joint_idx]['z']
                        ])
                    else:
                        frame_features.extend([0.0, 0.0, 0.0])
                features.append(np.array(frame_features))
            return features
        
        student_features = extract_features(student_landmarks)
        standard_features = extract_features(standard_landmarks)
        
        # 使用DTW找到最佳对应关系
        distance, path = fastdtw(student_features, standard_features, dist=euclidean)
        
        # 计算每个对应点的差异
        path_differences = []
        for student_idx, standard_idx in path:
            if (student_idx < len(student_landmarks) and 
                standard_idx < len(standard_landmarks)):
                diff = np.linalg.norm(student_features[student_idx] - standard_features[standard_idx])
                path_differences.append((diff, student_idx, standard_idx))
        
        # 按差异排序并选择差异最大的帧
        path_differences.sort(reverse=True, key=lambda x: x[0])
        
        # 选择差异大且相隔较远的帧
        selected_pairs = []
        used_student_frames = set()
        used_standard_frames = set()
        
        for diff, student_idx, standard_idx in path_differences:
            if len(selected_pairs) >= num_frames:
                break
            
            # 确保帧之间有足够间隔
            too_close = False
            for used_s, used_st in selected_pairs:
                if abs(student_idx - used_s) < 10 or abs(standard_idx - used_st) < 10:
                    too_close = True
                    break
            
            if not too_close:
                selected_pairs.append((student_idx, standard_idx))
                used_student_frames.add(student_idx)
                used_standard_frames.add(standard_idx)
        
        # 按学生帧索引排序
        selected_pairs.sort(key=lambda x: x[0])
        return selected_pairs
    
    def normalize_pose_to_student_frame(self, student_landmarks: List[Dict], 
                                      standard_landmarks: List[Dict],
                                      image_width: int, image_height: int) -> List[Dict]:
        """
        将标准动作归一化到学生动作的尺度和位置
        分别计算横向和纵向缩放因子以准确匹配学生动作比例
        增强的位置定位标准：
        1. 双肩中点与双髋中点连线的角度对齐（角度为零）
        2. 双肩中点与双髋中点连线的中点重合
        """
        if (student_landmarks is None or len(student_landmarks) == 0 or 
            standard_landmarks is None or len(standard_landmarks) == 0):
            return standard_landmarks
        
        # 获取关键参考点
        student_nose = self._get_nose_point(student_landmarks)
        student_shoulder_center = self._get_shoulder_center(student_landmarks)
        student_hip_center = self._get_hip_center(student_landmarks)
        
        standard_nose = self._get_nose_point(standard_landmarks)
        standard_shoulder_center = self._get_shoulder_center(standard_landmarks)
        standard_hip_center = self._get_hip_center(standard_landmarks)
        
        if (student_nose is None or student_shoulder_center is None or student_hip_center is None or 
            standard_nose is None or standard_shoulder_center is None or standard_hip_center is None):
            return standard_landmarks
        
        # 计算纵向缩放因子（基于身体高度比例）
        # 1. 鼻子到双肩中点的距离
        student_nose_to_shoulder = np.linalg.norm(student_nose[:2] - student_shoulder_center[:2])
        standard_nose_to_shoulder = np.linalg.norm(standard_nose[:2] - standard_shoulder_center[:2])
        
        # 2. 双肩中点到双髋中点的距离
        student_shoulder_to_hip = np.linalg.norm(student_shoulder_center[:2] - student_hip_center[:2])
        standard_shoulder_to_hip = np.linalg.norm(standard_shoulder_center[:2] - standard_hip_center[:2])
        
        # 纵向缩放因子（使用身体高度比例）
        nose_shoulder_scale = (student_nose_to_shoulder / standard_nose_to_shoulder 
                              if standard_nose_to_shoulder > 0 else 1.0)
        shoulder_hip_scale = (student_shoulder_to_hip / standard_shoulder_to_hip 
                             if standard_shoulder_to_hip > 0 else 1.0)
        
        scale_y = (nose_shoulder_scale + shoulder_hip_scale) / 2
        
        # 计算横向缩放因子（基于肩宽和髋宽比例）
        # 获取肩宽
        student_shoulder_width = self._get_shoulder_width(student_landmarks)
        standard_shoulder_width = self._get_shoulder_width(standard_landmarks)
        
        # 获取髋宽
        student_hip_width = self._get_hip_width(student_landmarks)
        standard_hip_width = self._get_hip_width(standard_landmarks)
        
        # 计算横向缩放因子
        shoulder_width_scale = (student_shoulder_width / standard_shoulder_width 
                               if standard_shoulder_width > 0 else 1.0)
        hip_width_scale = (student_hip_width / standard_hip_width 
                          if standard_hip_width > 0 else 1.0)
        
        # 使用肩宽和髋宽的平均值作为横向缩放因子
        scale_x = (shoulder_width_scale + hip_width_scale) / 2
        
        # 增强的位置定位标准实现
        # 1. 计算躯干角度差异并进行角度对齐
        student_torso_angle = self._calculate_torso_angle(student_shoulder_center, student_hip_center)
        standard_torso_angle = self._calculate_torso_angle(standard_shoulder_center, standard_hip_center)
        
        # 计算需要旋转的角度（使标准动作的躯干角度与学生动作一致）
        # 正确的计算方法：standard_angle - student_angle
        rotation_angle = standard_torso_angle - student_torso_angle
        
        # 2. 计算对齐参考点（使用躯干中心作为对齐基准，提高髋部位置精度）
        student_torso_center = (student_shoulder_center + student_hip_center) / 2
        standard_torso_center = (standard_shoulder_center + standard_hip_center) / 2
        
        student_reference = student_torso_center[:2]
        standard_reference = standard_torso_center[:2]
        
        # 归一化标准动作
        normalized_standard = []
        
        for i, landmark in enumerate(standard_landmarks):
            if landmark['visibility'] > 0.5:
                # 原始位置相对于标准动作参考点的偏移
                original_pos = np.array([landmark['x'], landmark['y']])
                offset = original_pos - standard_reference
                
                # 分别处理横向和纵向偏移，避免交叉影响
                # 横向偏移只使用横向缩放因子
                # 纵向偏移只使用纵向缩放因子
                scaled_offset_x = offset[0] * scale_x
                scaled_offset_y = offset[1] * scale_y
                scaled_offset = np.array([scaled_offset_x, scaled_offset_y])
                
                # 应用旋转变换
                if abs(rotation_angle) > 0.001:  # 避免不必要的旋转计算
                    rotated_offset = self._rotate_point_around_center(
                        scaled_offset, 
                        np.array([0, 0]),  # 绕原点旋转
                        rotation_angle
                    )
                    scaled_offset = rotated_offset
                
                # 重新定位到学生的参考点
                new_pos = student_reference + scaled_offset
                
                normalized_landmark = {
                    'x': new_pos[0],
                    'y': new_pos[1],
                    'z': landmark['z'] * scale_y,  # z坐标使用纵向缩放因子
                    'visibility': landmark['visibility']
                }
            else:
                normalized_landmark = landmark.copy()
            
            normalized_standard.append(normalized_landmark)
        
        return normalized_standard
    
    def _get_nose_point(self, landmarks: List[Dict]) -> Optional[np.ndarray]:
        """获取鼻子关键点"""
        if (len(landmarks) > 0 and landmarks[0]['visibility'] > 0.5):
            return np.array([landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']])
        return None
    
    def _rotate_point_around_center(self, point: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
        """
        将点绕中心点旋转指定角度
        
        Args:
            point: 要旋转的点 [x, y]
            center: 旋转中心 [x, y]
            angle: 旋转角度（弧度）
        
        Returns:
            旋转后的点坐标
        """
        # 将点平移到原点
        translated_point = point - center
        
        # 应用旋转矩阵
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
        
        rotated_point = rotation_matrix @ translated_point
        
        # 平移回原位置
        return rotated_point + center
    
    def _calculate_torso_angle(self, shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
        """
        计算躯干角度（肩髋连线与垂直方向的夹角）
        
        Args:
            shoulder_center: 双肩中点
            hip_center: 双髋中点
            
        Returns:
            角度（弧度），正值表示向右倾斜，负值表示向左倾斜
        """
        # 计算肩髋连线向量（从肩到髋）
        torso_vector = hip_center[:2] - shoulder_center[:2]
        
        # 计算与垂直向下方向（0, 1）的夹角
        # 使用atan2计算角度，注意参数顺序：atan2(x, y)表示向量(x,y)与正y轴的夹角
        angle = np.arctan2(torso_vector[0], torso_vector[1])
        
        return angle
    
    def _get_shoulder_center(self, landmarks: List[Dict]) -> Optional[np.ndarray]:
        """获取双肩中点"""
        if (len(landmarks) > 12 and 
            landmarks[11]['visibility'] > 0.5 and 
            landmarks[12]['visibility'] > 0.5):
            left_shoulder = np.array([landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']])
            right_shoulder = np.array([landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']])
            return (left_shoulder + right_shoulder) / 2
        return None
    
    def _get_hip_center(self, landmarks: List[Dict]) -> Optional[np.ndarray]:
        """获取双髋中点"""
        if (len(landmarks) > 24 and 
            landmarks[23]['visibility'] > 0.5 and 
            landmarks[24]['visibility'] > 0.5):
            left_hip = np.array([landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']])
            right_hip = np.array([landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z']])
            return (left_hip + right_hip) / 2
        return None
    
    def _get_shoulder_width(self, landmarks: List[Dict]) -> float:
        """获取肩宽"""
        left_shoulder = landmarks[11] if len(landmarks) > 11 else None
        right_shoulder = landmarks[12] if len(landmarks) > 12 else None
        
        if (left_shoulder and right_shoulder and 
            left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5):
            left_pos = np.array([left_shoulder['x'], left_shoulder['y']])
            right_pos = np.array([right_shoulder['x'], right_shoulder['y']])
            return np.linalg.norm(right_pos - left_pos)
        return 0.0
    
    def _get_hip_width(self, landmarks: List[Dict]) -> float:
        """获取髋宽"""
        left_hip = landmarks[23] if len(landmarks) > 23 else None
        right_hip = landmarks[24] if len(landmarks) > 24 else None
        
        if (left_hip and right_hip and 
            left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5):
            left_pos = np.array([left_hip['x'], left_hip['y']])
            right_pos = np.array([right_hip['x'], right_hip['y']])
            return np.linalg.norm(right_pos - left_pos)
        return 0.0
    
    def draw_pose_on_image(self, image: np.ndarray, landmarks: List[Dict], 
                          color: Tuple[int, int, int], thickness: int = 3) -> np.ndarray:
        """
        在图像上绘制姿态骨骼点，确保坐标准确映射
        """
        if landmarks is None or len(landmarks) == 0:
            return image
        
        h, w = image.shape[:2]
        
        # 绘制连接线
        for connection in self.pose_connections:
            start_idx, end_idx = connection
            
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx]['visibility'] > 0.5 and
                landmarks[end_idx]['visibility'] > 0.5):
                
                # 确保坐标在有效范围内
                start_x = max(0, min(w-1, int(landmarks[start_idx]['x'] * w)))
                start_y = max(0, min(h-1, int(landmarks[start_idx]['y'] * h)))
                end_x = max(0, min(w-1, int(landmarks[end_idx]['x'] * w)))
                end_y = max(0, min(h-1, int(landmarks[end_idx]['y'] * h)))
                
                start_point = (start_x, start_y)
                end_point = (end_x, end_y)
                
                cv2.line(image, start_point, end_point, color, thickness)
        
        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            if landmark['visibility'] > 0.5:
                # 确保坐标在有效范围内
                x = max(0, min(w-1, int(landmark['x'] * w)))
                y = max(0, min(h-1, int(landmark['y'] * h)))
                point = (x, y)
                cv2.circle(image, point, thickness + 2, color, -1)
        
        return image
    
    def create_comparison_visualization(self, student_video_path: str,
                                     student_landmarks: List[Dict],
                                     standard_landmarks: List[Dict],
                                     output_dir: str,
                                     frame_interval: int = None) -> List[str]:
        """
        创建精确的动作比较可视化图像
        
        参数:
        frame_interval: 采样间隔，如果为None则自动计算
        """
        # 使用DTW选择关键帧对
        key_frame_pairs = self.select_key_frames_with_dtw(student_landmarks, standard_landmarks)
        
        if not key_frame_pairs:
            print("警告: 未能选择到有效的关键帧")
            return []
        
        # 打开视频文件
        cap = cv2.VideoCapture(student_video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {student_video_path}")
        
        # 获取视频尺寸和参数
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 如果没有提供frame_interval，则根据视频FPS自动计算
        if frame_interval is None:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = 10.0  # 与extract_landmarks中的默认值保持一致
            frame_interval = max(1, int(original_fps / target_fps)) if original_fps > 0 else 3
        
        output_paths = []
        
        for i, (student_frame_idx, standard_frame_idx) in enumerate(key_frame_pairs):
            # 读取对应帧的视频画面
            # 将骨骼点索引映射到实际视频帧索引
            # 使用动态计算的frame_interval而非固定的3
            actual_video_frame_idx = student_frame_idx * frame_interval
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_video_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"警告: 无法读取第{actual_video_frame_idx}帧 (骨骼点索引: {student_frame_idx})")
                continue
            
            # 获取对应帧的关键点数据
            if (student_frame_idx >= len(student_landmarks) or 
                standard_frame_idx >= len(standard_landmarks)):
                continue
            
            student_frame_landmarks = student_landmarks[student_frame_idx]
            standard_frame_landmarks = standard_landmarks[standard_frame_idx]
            
            if (student_frame_landmarks is None or len(student_frame_landmarks) == 0 or 
                standard_frame_landmarks is None or len(standard_frame_landmarks) == 0):
                continue
            
            # 归一化标准动作到学生动作的尺度
            normalized_standard = self.normalize_pose_to_student_frame(
                student_frame_landmarks, standard_frame_landmarks, frame_width, frame_height
            )
            
            # 创建可视化图像
            result_image = frame.copy()
            
            # 绘制学生动作（红色）
            result_image = self.draw_pose_on_image(
                result_image, student_frame_landmarks, (0, 0, 255), thickness=18
            )
            
            # 绘制标准动作（绿色，在上层）
            result_image = self.draw_pose_on_image(
                result_image, normalized_standard, (0, 255, 0), thickness=20
            )
            
            # 添加文字说明
            cv2.putText(result_image, f"Student Frame {student_frame_idx} (Video: {actual_video_frame_idx}) vs Standard Frame {standard_frame_idx}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_image, "Red: Student, Green: Standard", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存图像
            output_path = os.path.join(output_dir, f"comparison_frame_{student_frame_idx:03d}_vs_{standard_frame_idx:03d}.png")
            cv2.imwrite(output_path, result_image)
            output_paths.append(output_path)
            
            print(f"已生成精确比较图像: {output_path}")
        
        cap.release()
        return output_paths


def create_pose_comparison_visualization(student_video_path: str,
                                       student_landmarks: List[Dict],
                                       standard_landmarks: List[Dict],
                                       output_dir: str = "results",
                                       frame_interval: int = None) -> List[str]:
    """
    创建姿态比较可视化的便捷函数
    
    参数:
    frame_interval: 采样间隔，如果为None则自动计算
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化器
    visualizer = PoseVisualizer()
    
    # 生成可视化图像
    return visualizer.create_comparison_visualization(
        student_video_path, student_landmarks, standard_landmarks, output_dir, frame_interval
    )