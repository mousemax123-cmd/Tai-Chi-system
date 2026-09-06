# -*- coding: utf-8 -*-
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
# 移除sklearn依赖，使用numpy实现
import matplotlib.pyplot as plt
import os
import matplotlib

# 全局校准器变量 - 移到文件开头
_global_calibrator = None

# 导入数据预处理函数
try:
    from data_process import interpolate_missing_landmarks, smooth_landmarks
except ImportError:
    def interpolate_missing_landmarks(landmarks_data, visibility_threshold=0.5):
        return landmarks_data
    
    def smooth_landmarks(landmarks_data, window_size=5):
        return landmarks_data

# 设置matplotlib支持中文显示
try:
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("警告：无法设置中文字体，图表中的中文可能无法正确显示")

class ImprovedTaichiScorer:
    """改进的太极拳动作评分系统"""
    
    def __init__(self):
        # 关键关节索引
        self.key_joints = {
            'shoulders': [11, 12],  # 左肩、右肩
            'elbows': [13, 14],     # 左肘、右肘
            'wrists': [15, 16],     # 左腕、右腕
            'hips': [23, 24],       # 左髋、右髋
            'knees': [25, 26],      # 左膝、右膝
            'ankles': [27, 28]      # 左踝、右踝
        }
        
        # 特征权重
        self.feature_weights = {
            'position': 0.4,        # 位置特征权重
            'velocity': 0.25,       # 速度特征权重
            'angles': 0.25,         # 角度特征权重
            'symmetry': 0.1         # 对称性特征权重
        }
    
    def normalize_landmarks_robust(self, landmarks):
        """鲁棒的关键点标准化"""
        normalized_data = []
        
        for frame in landmarks:
            # 计算髋部中心
            left_hip = np.array([frame[23]['x'], frame[23]['y'], frame[23]['z']])
            right_hip = np.array([frame[24]['x'], frame[24]['y'], frame[24]['z']])
            hip_center = (left_hip + right_hip) / 2
            
            # 计算肩部中心
            left_shoulder = np.array([frame[11]['x'], frame[11]['y'], frame[11]['z']])
            right_shoulder = np.array([frame[12]['x'], frame[12]['y'], frame[12]['z']])
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            # 计算身体比例尺度（使用多个参考距离的平均值）
            torso_height = np.linalg.norm(shoulder_center - hip_center)
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            hip_width = np.linalg.norm(right_hip - left_hip)
            
            # 使用多个尺度的平均值，提高鲁棒性
            scale = np.mean([torso_height, shoulder_width, hip_width])
            scale = max(scale, 0.1)  # 防止除零
            
            # 标准化每个关键点
            normalized_frame = []
            for landmark in frame:
                point = np.array([landmark['x'], landmark['y'], landmark['z']])
                normalized_point = (point - hip_center) / scale
                normalized_frame.append({
                    'x': normalized_point[0],
                    'y': normalized_point[1], 
                    'z': normalized_point[2],
                    'visibility': landmark['visibility']
                })
            
            normalized_data.append(normalized_frame)
        
        return normalized_data
    
    def extract_position_features(self, landmarks, return_visibility_mask=False):
        """提取位置特征，支持可见度加权"""
        features = []
        visibility_masks = []
        
        for frame in landmarks:
            frame_features = []
            frame_visibility = []
            
            # 提取关键关节位置
            for joint_group in self.key_joints.values():
                for joint_idx in joint_group:
                    joint = frame[joint_idx]
                    visibility = joint['visibility']
                    
                    # 使用可见度加权的特征值
                    frame_features.extend([joint['x'], joint['y'], joint['z']])
                    frame_visibility.extend([visibility, visibility, visibility])
            
            features.append(frame_features)
            visibility_masks.append(frame_visibility)
        
        if return_visibility_mask:
            return np.array(features), np.array(visibility_masks)
        else:
            return np.array(features)
    
    def extract_velocity_features(self, landmarks):
        """提取速度特征"""
        position_features = self.extract_position_features(landmarks)
        
        if len(position_features) < 2:
            return np.zeros((len(position_features), position_features.shape[1]))
        
        # 计算帧间速度
        velocities = []
        velocities.append(np.zeros(position_features.shape[1]))  # 第一帧速度为0
        
        for i in range(1, len(position_features)):
            velocity = position_features[i] - position_features[i-1]
            velocities.append(velocity)
        
        return np.array(velocities)
    
    def extract_angle_features(self, landmarks):
        """提取角度特征"""
        angles_data = []
        
        for frame in landmarks:
            frame_angles = []
            
            # 右肘角度
            try:
                right_shoulder = np.array([frame[12]['x'], frame[12]['y'], frame[12]['z']])
                right_elbow = np.array([frame[14]['x'], frame[14]['y'], frame[14]['z']])
                right_wrist = np.array([frame[16]['x'], frame[16]['y'], frame[16]['z']])
                
                v1 = right_shoulder - right_elbow
                v2 = right_wrist - right_elbow
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                                                   (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                frame_angles.append(angle)
            except:
                frame_angles.append(90)  # 默认角度
            
            # 左肘角度
            try:
                left_shoulder = np.array([frame[11]['x'], frame[11]['y'], frame[11]['z']])
                left_elbow = np.array([frame[13]['x'], frame[13]['y'], frame[13]['z']])
                left_wrist = np.array([frame[15]['x'], frame[15]['y'], frame[15]['z']])
                
                v1 = left_shoulder - left_elbow
                v2 = left_wrist - left_elbow
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                                                   (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                frame_angles.append(angle)
            except:
                frame_angles.append(90)
            
            # 右膝角度
            try:
                right_hip = np.array([frame[24]['x'], frame[24]['y'], frame[24]['z']])
                right_knee = np.array([frame[26]['x'], frame[26]['y'], frame[26]['z']])
                right_ankle = np.array([frame[28]['x'], frame[28]['y'], frame[28]['z']])
                
                v1 = right_hip - right_knee
                v2 = right_ankle - right_knee
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                                                   (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                frame_angles.append(angle)
            except:
                frame_angles.append(180)
            
            # 左膝角度
            try:
                left_hip = np.array([frame[23]['x'], frame[23]['y'], frame[23]['z']])
                left_knee = np.array([frame[25]['x'], frame[25]['y'], frame[25]['z']])
                left_ankle = np.array([frame[27]['x'], frame[27]['y'], frame[27]['z']])
                
                v1 = left_hip - left_knee
                v2 = left_ankle - left_knee
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                                                   (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
                frame_angles.append(angle)
            except:
                frame_angles.append(180)
            
            angles_data.append(frame_angles)
        
        return np.array(angles_data)
    
    def extract_symmetry_features(self, landmarks):
        """提取对称性特征"""
        symmetry_data = []
        
        for frame in landmarks:
            symmetry_features = []
            
            # 左右肩高度差
            left_shoulder_y = frame[11]['y']
            right_shoulder_y = frame[12]['y']
            shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
            symmetry_features.append(shoulder_diff)
            
            # 左右髋高度差
            left_hip_y = frame[23]['y']
            right_hip_y = frame[24]['y']
            hip_diff = abs(left_hip_y - right_hip_y)
            symmetry_features.append(hip_diff)
            
            # 左右手腕高度差
            left_wrist_y = frame[15]['y']
            right_wrist_y = frame[16]['y']
            wrist_diff = abs(left_wrist_y - right_wrist_y)
            symmetry_features.append(wrist_diff)
            
            symmetry_data.append(symmetry_features)
        
        return np.array(symmetry_data)
    
    def extract_comprehensive_features(self, landmarks, return_visibility_mask=False):
        """提取综合特征，支持可见度掩码"""
        # 标准化关键点
        normalized_landmarks = self.normalize_landmarks_robust(landmarks)
        
        # 提取各类特征
        if return_visibility_mask:
            position_features, position_mask = self.extract_position_features(normalized_landmarks, return_visibility_mask=True)
        else:
            position_features = self.extract_position_features(normalized_landmarks)
            
        velocity_features = self.extract_velocity_features(normalized_landmarks)
        angle_features = self.extract_angle_features(normalized_landmarks)
        symmetry_features = self.extract_symmetry_features(normalized_landmarks)
        
        # 改进的特征标准化，对相同动作更鲁棒
        def standardize_features_robust(features):
            if len(features) <= 1:
                return features
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            # 使用更大的最小标准差，避免过度标准化
            min_std = 0.01  # 增加最小标准差
            std = np.where(std < min_std, min_std, std)
            return (features - mean) / std
        
        # 只对变化较大的特征进行标准化
        if len(position_features) > 1:
            # 检查特征变化程度，如果变化很小则不进行标准化
            pos_std = np.std(position_features, axis=0)
            if np.mean(pos_std) > 0.001:  # 只有变化足够大时才标准化
                position_features = standardize_features_robust(position_features)
            
            vel_std = np.std(velocity_features, axis=0)
            if np.mean(vel_std) > 0.001:
                velocity_features = standardize_features_robust(velocity_features)
            
            ang_std = np.std(angle_features, axis=0)
            if np.mean(ang_std) > 0.1:  # 角度特征使用更大的阈值
                angle_features = standardize_features_robust(angle_features)
            
            sym_std = np.std(symmetry_features, axis=0)
            if np.mean(sym_std) > 0.001:
                symmetry_features = standardize_features_robust(symmetry_features)
        
        # 组合所有特征
        combined_features = {
            'position': position_features,
            'velocity': velocity_features,
            'angles': angle_features,
            'symmetry': symmetry_features
        }
        
        if return_visibility_mask:
            # 为其他特征创建掩码（基于位置特征的掩码）
            combined_masks = {
                'position': position_mask,
                'velocity': position_mask,  # 速度基于位置，使用相同掩码
                'angles': np.ones_like(angle_features),  # 角度特征暂时使用全1掩码
                'symmetry': np.ones_like(symmetry_features)  # 对称性特征使用全1掩码
            }
            return combined_features, combined_masks
        else:
            return combined_features
    
    def calculate_dtw_score(self, student_features, standard_features, feature_type, 
                           student_mask=None, standard_mask=None):
        """计算DTW评分，支持可见度掩码，针对长视频进行了性能优化"""
        
        # --- 性能优化：自动降采样 ---
        # 如果帧数过多，自动进行降采样以减少计算量
        # 目标是将计算矩阵限制在可接受的大小（例如 500x500 以内）
        n_student_raw = len(student_features)
        n_standard_raw = len(standard_features)
        
        # 计算降采样步长 (Stride)
        # 保持至少 200 帧用于计算，或者最大步长为 5
        max_frames = 400
        stride = max(1, min(n_student_raw, n_standard_raw) // max_frames)
        
        if stride > 1:
            # print(f"  [性能优化] 检测到长视频，启用降采样 (步长: {stride})")
            # 对特征和掩码进行切片降采样
            s_feat = student_features[::stride]
            std_feat = standard_features[::stride]
            s_mask = student_mask[::stride] if student_mask is not None else None
            std_mask = standard_mask[::stride] if standard_mask is not None else None
        else:
            s_feat = student_features
            std_feat = standard_features
            s_mask = student_mask
            std_mask = standard_mask

        # 定义可见度加权距离函数 (保持不变)
        def visibility_weighted_distance(a, b, mask_a=None, mask_b=None):
            if mask_a is None or mask_b is None:
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0: return 1.0
                return 1 - np.dot(a, b) / (norm_a * norm_b)
            
            combined_mask = np.minimum(mask_a, mask_b)
            visibility_threshold = 0.5
            valid_dims = combined_mask > visibility_threshold
            
            if not np.any(valid_dims): return 1.0
            
            a_valid = a[valid_dims]
            b_valid = b[valid_dims]
            weights = combined_mask[valid_dims]
            
            weighted_diff = (a_valid - b_valid) * weights
            distance = np.sqrt(np.sum(weighted_diff ** 2))
            
            max_possible_distance = np.sqrt(np.sum(weights ** 2) * 2)
            if max_possible_distance > 0:
                distance = distance / max_possible_distance
            
            return min(1.0, distance)
        
        # 计算DTW距离
        if s_mask is not None and std_mask is not None:
            n_student = len(s_feat)
            n_standard = len(std_feat)
            
            # --- 性能优化：使用窗口限制 (Window constraint) ---
            # 假设动作大体同步，只计算对角线附近的距离
            # 窗口大小设为较大序列长度的 20%
            window_size = max(n_student, n_standard) * 0.2
            
            dtw_matrix = np.full((n_student + 1, n_standard + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            # 计算 DTW (带窗口限制)
            for i in range(1, n_student + 1):
                # 计算 j 的起止范围，只遍历对角线附近的点
                j_start = max(1, int(i - window_size))
                j_end = min(n_standard + 1, int(i + window_size))
                
                for j in range(j_start, j_end):
                    dist = visibility_weighted_distance(
                        s_feat[i-1], std_feat[j-1],
                        s_mask[i-1], std_mask[j-1]
                    )
                    
                    dtw_matrix[i, j] = dist + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            distance = dtw_matrix[n_student, n_standard]
            
            # 如果由于窗口限制导致无穷大（极不可能），回退到非无穷大值
            if np.isinf(distance):
                distance = 100.0 # 惩罚分
            
            path_length = max(n_student, n_standard) # 近似路径长度
            
            # 模拟路径 (为了兼容接口，不需要真实回溯庞大的路径)
            path = [] 

        else:
            # 对于无掩码情况（如角度），直接使用 fastdtw (它内部已经优化)
            # 但同样使用降采样后的数据
            from fastdtw import fastdtw
            def cosine_distance(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0: return 1.0
                return 1 - np.dot(a, b) / (norm_a * norm_b)
            
            distance, path = fastdtw(s_feat, std_feat, dist=cosine_distance)
            path_length = len(path)

        # 归一化距离
        normalized_distance = distance / path_length if path_length > 0 else 1.0
        
        # 优化的评分转换函数 (保持不变)
        if normalized_distance < 0.05: score = 100 - normalized_distance * 20
        elif normalized_distance < 0.15: score = 99 - (normalized_distance - 0.05) * 40
        elif normalized_distance < 0.35: score = 95 - (normalized_distance - 0.15) * 75
        elif normalized_distance < 0.65: score = 80 - (normalized_distance - 0.35) * 83.33
        else: score = max(0, 55 - (normalized_distance - 0.65) * 78.57)
        
        return max(0, min(100, score)), [] # 返回空路径以节省内存
    def calculate_correlation_score(self, student_features, standard_features):
        """计算相关性评分"""
        if len(student_features) == 0 or len(standard_features) == 0:
            return 0
        
        # 将特征序列展平
        student_flat = student_features.flatten()
        standard_flat = standard_features.flatten()
        
        # 调整长度
        min_len = min(len(student_flat), len(standard_flat))
        student_flat = student_flat[:min_len]
        standard_flat = standard_flat[:min_len]
        
        # 计算皮尔逊相关系数
        try:
            correlation, _ = pearsonr(student_flat, standard_flat)
            # 优化相关性评分转换，对高相关性更宽容
            if correlation > 0.95:  # 极高相关性（几乎完全相同）
                score = 99 + (correlation - 0.95) * 20  # 99-100分
            elif correlation > 0.85:  # 很高相关性
                score = 95 + (correlation - 0.85) * 40  # 95-99分
            elif correlation > 0.7:  # 高相关性
                score = 85 + (correlation - 0.7) * 66.67  # 85-95分
            elif correlation > 0.5:  # 中等相关性
                score = 70 + (correlation - 0.5) * 75  # 70-85分
            elif correlation > 0:  # 低相关性
                score = correlation * 140  # 0-70分
            else:
                score = 0
            return min(100, max(0, score))
        except:
            return 0
    
    def score_movement_improved(self, student_landmarks, standard_landmarks):
        """改进的动作评分函数，支持可见度掩码"""
        print("开始改进的动作评分...")
        
        # 数据预处理
        student_landmarks = interpolate_missing_landmarks(student_landmarks, visibility_threshold=0.6)
        student_landmarks = smooth_landmarks(student_landmarks, window_size=3)
        
        standard_landmarks = interpolate_missing_landmarks(standard_landmarks, visibility_threshold=0.6)
        standard_landmarks = smooth_landmarks(standard_landmarks, window_size=3)
        
        # 提取特征和可见度掩码
        student_features, student_masks = self.extract_comprehensive_features(student_landmarks, return_visibility_mask=True)
        standard_features, standard_masks = self.extract_comprehensive_features(standard_landmarks, return_visibility_mask=True)
        
        # 计算各类特征的评分
        scores = {}
        
        # 位置特征评分（使用可见度掩码）
        pos_dtw_score, pos_path = self.calculate_dtw_score(
            student_features['position'], standard_features['position'], 'position',
            student_masks['position'], standard_masks['position']
        )
        pos_corr_score = self.calculate_correlation_score(
            student_features['position'], standard_features['position']
        )
        scores['position'] = {
            'dtw': pos_dtw_score,
            'correlation': pos_corr_score,
            'combined': (pos_dtw_score * 0.7 + pos_corr_score * 0.3)
        }
        
        # 速度特征评分（使用可见度掩码）
        vel_dtw_score, vel_path = self.calculate_dtw_score(
            student_features['velocity'], standard_features['velocity'], 'velocity',
            student_masks['velocity'], standard_masks['velocity']
        )
        vel_corr_score = self.calculate_correlation_score(
            student_features['velocity'], standard_features['velocity']
        )
        scores['velocity'] = {
            'dtw': vel_dtw_score,
            'correlation': vel_corr_score,
            'combined': (vel_dtw_score * 0.7 + vel_corr_score * 0.3)
        }
        
        # 角度特征评分（暂不使用掩码，因为角度计算复杂）
        ang_dtw_score, ang_path = self.calculate_dtw_score(
            student_features['angles'], standard_features['angles'], 'angles'
        )
        ang_corr_score = self.calculate_correlation_score(
            student_features['angles'], standard_features['angles']
        )
        scores['angles'] = {
            'dtw': ang_dtw_score,
            'correlation': ang_corr_score,
            'combined': (ang_dtw_score * 0.7 + ang_corr_score * 0.3)
        }
        
        # 对称性特征评分
        sym_dtw_score, sym_path = self.calculate_dtw_score(
            student_features['symmetry'], standard_features['symmetry'], 'symmetry'
        )
        sym_corr_score = self.calculate_correlation_score(
            student_features['symmetry'], standard_features['symmetry']
        )
        scores['symmetry'] = {
            'dtw': sym_dtw_score,
            'correlation': sym_corr_score,
            'combined': (sym_dtw_score * 0.7 + sym_corr_score * 0.3)
        }
        
        # 计算加权总分
        total_score = (
            scores['position']['combined'] * self.feature_weights['position'] +
            scores['velocity']['combined'] * self.feature_weights['velocity'] +
            scores['angles']['combined'] * self.feature_weights['angles'] +
            scores['symmetry']['combined'] * self.feature_weights['symmetry']
        )
        
        # 详细错误分析
        errors = self.analyze_errors_detailed(scores, student_features, standard_features)
        
        print(f"评分完成，总分: {total_score:.2f}")
        
        return {
            'total_score': total_score,
            'detailed_scores': scores,
            'errors': errors,
            'features': {
                'student': student_features,
                'standard': standard_features
            },
            'masks': {
                'student': student_masks,
                'standard': standard_masks
            }
        }
    
    def analyze_errors_detailed(self, scores, student_features, standard_features):
        """详细的错误分析"""
        errors = []
        
        feature_names = {
            'position': '身体位置',
            'velocity': '动作速度',
            'angles': '关节角度', 
            'symmetry': '身体对称性'
        }
        
        for feature_type, score_dict in scores.items():
            feature_name = feature_names.get(feature_type, feature_type)
            # 获取综合分数
            combined_score = score_dict.get('combined', 0) if isinstance(score_dict, dict) else score_dict
            
            if combined_score < 60:
                severity = "严重偏差"
            elif combined_score < 75:
                severity = "中度偏差"
            elif combined_score < 85:
                severity = "轻微偏差"
            else:
                severity = "良好"
            
            errors.append({
                'feature': feature_name,
                'score': combined_score,
                'severity': severity
            })
        
        return errors

# 改进的评分函数（保持与原接口兼容）
def score_movement_improved(student_landmarks, standard_landmarks, weights=None):
    """改进的评分函数，保持与原接口兼容"""
    scorer = ImprovedTaichiScorer()
    return scorer.score_movement_improved(student_landmarks, standard_landmarks)

# 改进建议函数
def get_error_suggestions_improved(errors):
    """根据错误分析提供改进建议"""
    suggestions = []
    
    suggestion_map = {
        '身体位置': [
            "注意保持身体重心稳定，避免过度前倾或后仰",
            "确保动作幅度适中，不要过大或过小",
            "保持身体各部位协调配合"
        ],
        '动作速度': [
            "控制动作节奏，保持匀速缓慢的太极拳特色",
            "避免动作过快或过慢，保持连贯性",
            "注意动作间的停顿和过渡"
        ],
        '关节角度': [
            "注意手臂弯曲角度，保持自然圆润",
            "膝盖弯曲要适度，避免过直或过弯",
            "保持关节放松，避免僵硬"
        ],
        '身体对称性': [
            "注意左右身体的平衡，避免偏向一侧",
            "保持肩膀水平，避免高低不平",
            "确保左右手动作对称协调"
        ]
    }
    
    for error in errors:
        if error['severity'] in ['严重偏差', '中度偏差']:
            feature = error['feature']
            if feature in suggestion_map:
                suggestions.extend(suggestion_map[feature])
    
    # 去重并限制建议数量
    suggestions = list(set(suggestions))[:5]
    
    if not suggestions:
        suggestions = ["整体动作表现良好，继续保持！"]
    
    return suggestions


class AdaptiveScoringCalibrator:
    """自适应评分校准器 - 基于Z-score标准化的统计校准法"""
    
    def __init__(self, target_mean=75.0, target_std=12.0):
        """
        初始化校准器
        
        Args:
            target_mean: 目标平均分
            target_std: 目标标准差
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.calibration_params = None
        self.is_calibrated = False
    
    def fit(self, scores):
        """
        基于历史分数拟合校准参数
        
        Args:
            scores: 历史分数列表或数组
        """
        scores = np.array(scores)
        
        # 计算原始分数的统计参数
        original_mean = np.mean(scores)
        original_std = np.std(scores)
        
        # 避免标准差为0的情况
        if original_std == 0:
            original_std = 1.0
        
        # 保存校准参数
        self.calibration_params = {
            'original_mean': original_mean,
            'original_std': original_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }
        
        self.is_calibrated = True
        
        return self
    
    def transform(self, scores):
        """
        应用校准变换
        
        Args:
            scores: 需要校准的分数
            
        Returns:
            校准后的分数
        """
        if not self.is_calibrated:
            raise ValueError("校准器尚未拟合，请先调用fit()方法")
        
        scores = np.array(scores)
        params = self.calibration_params
        
        # Z-score标准化 + 重映射
        z_scores = (scores - params['original_mean']) / params['original_std']
        calibrated_scores = z_scores * params['target_std'] + params['target_mean']
        
        # 确保分数在合理范围内 [0, 100]
        calibrated_scores = np.clip(calibrated_scores, 0, 100)
        
        return calibrated_scores
    
    def fit_transform(self, scores):
        """
        拟合并变换分数
        
        Args:
            scores: 分数列表或数组
            
        Returns:
            校准后的分数
        """
        return self.fit(scores).transform(scores)
    
    def get_calibration_info(self):
        """获取校准信息"""
        if not self.is_calibrated:
            return None
        
        return {
            'method': 'statistical_calibration',
            'original_mean': self.calibration_params['original_mean'],
            'original_std': self.calibration_params['original_std'],
            'target_mean': self.calibration_params['target_mean'],
            'target_std': self.calibration_params['target_std'],
            'is_calibrated': self.is_calibrated
        }

def initialize_scoring_calibration(historical_scores=None, target_mean=75.0, target_std=12.0):
    """
    初始化全局评分校准器

    Args:
        historical_scores: 历史分数数据，用于拟合校准参数
        target_mean: 目标平均分
        target_std: 目标标准差
        
    Returns:
        tuple: (校准器对象, 是否成功设置全局变量)
    """
    global _global_calibrator

    # 创建校准器实例
    calibrator = AdaptiveScoringCalibrator(target_mean=target_mean, target_std=target_std)

    # 如果没有提供历史数据，使用模拟的典型评分分布来训练校准器
    if historical_scores is None:
        # 生成模拟的历史评分数据，模拟严格评分系统的典型分布
        # 假设原始评分系统较为严格，平均分约55分，标准差约15分
        np.random.seed(42)  # 确保结果可重现
        historical_scores = np.random.normal(55.0, 15.0, 100)
        # 限制分数范围在0-100之间
        historical_scores = np.clip(historical_scores, 0, 100)

    # 训练校准器
    calibrator.fit(historical_scores)
    
    # 设置全局变量
    _global_calibrator = calibrator
    
    # 验证全局变量是否正确设置
    global_set_success = _global_calibrator is not None and _global_calibrator.is_calibrated
    
    return calibrator, global_set_success

def get_global_calibrator():
    """获取全局校准器实例"""
    global _global_calibrator
    return _global_calibrator

def is_calibration_initialized():
    """检查校准系统是否已初始化"""
    global _global_calibrator
    return _global_calibrator is not None and _global_calibrator.is_calibrated

def apply_score_calibration(score):
    """
    应用评分校准
    
    Args:
        score: 原始评分
        
    Returns:
        校准后的评分
    """
    global _global_calibrator
    
    if _global_calibrator is None or not _global_calibrator.is_calibrated:
        # 如果校准器未初始化，返回原始分数
        return score
    
    try:
        calibrated_score = _global_calibrator.transform([score])[0]
        return float(calibrated_score)
    except Exception as e:
        print(f"评分校准失败: {e}")
        return score

def score_movement_improved_with_calibration(student_landmarks, standard_landmarks, weights=None, enable_calibration=True):
    """
    带校准功能的改进评分函数
    
    Args:
        student_landmarks: 学生动作关键点
        standard_landmarks: 标准动作关键点
        weights: 特征权重
        enable_calibration: 是否启用校准
        
    Returns:
        评分结果（包含校准后的分数）
    """
    # 获取原始评分
    result = score_movement_improved(student_landmarks, standard_landmarks, weights)
    
    if enable_calibration:
        # 应用校准
        original_score = result['total_score']
        calibrated_score = apply_score_calibration(original_score)
        
        # 更新结果
        result['original_score'] = original_score
        result['total_score'] = calibrated_score
        result['calibration_applied'] = True
        
        # 获取校准信息
        global _global_calibrator
        if _global_calibrator is not None:
            result['calibration_info'] = _global_calibrator.get_calibration_info()
    else:
        result['calibration_applied'] = False
    
    return result