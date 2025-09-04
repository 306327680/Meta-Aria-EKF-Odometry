#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Aria GPS/IMU EKF Fusion for Odometry

功能：
1) 读取 Project Aria VRS 文件中的 GPS 和 IMU 数据
2) 实现扩展卡尔曼滤波器 (EKF) 进行 GPS/IMU 融合
3) 生成改进的里程计轨迹
4) 对比显示 EKF 前后的位置数据

状态向量: [x, y, z, vx, vy, vz, roll, pitch, yaw]
- 位置: x, y, z (米)
- 速度: vx, vy, vz (米/秒)
- 姿态: roll, pitch, yaw (弧度)

依赖：
  pip install projectaria-tools matplotlib numpy scipy

作者: Claude Code Assistant
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import warnings

from projectaria_tools.core import data_provider

# 忽略一些常见的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


class EKF:
    """扩展卡尔曼滤波器用于GPS/IMU融合"""
    
    def __init__(self, imu_to_world_rotation=None, gyro_bias=None):
        # 状态向量 [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.state = np.zeros(9)
        
        # 状态协方差矩阵
        self.P = np.eye(9) * 0.1
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(9)
        self.Q[0:3, 0:3] *= 0.01   # 位置过程噪声
        self.Q[3:6, 3:6] *= 0.1    # 速度过程噪声
        self.Q[6:9, 6:9] *= 0.01   # 姿态过程噪声
        
        # GPS观测噪声协方差
        self.R_gps = np.eye(3) * 1.0  # GPS位置观测噪声
        
        # 重力向量将在外参估计后设置
        self.gravity = np.array([0, 0, 9.81])  # 默认值，将被更新
        
        # 使用预估计的IMU到世界坐标系转换
        self.imu_to_world_rotation = imu_to_world_rotation if imu_to_world_rotation is not None else np.eye(3)
        
        # 使用预估计的陀螺仪偏置
        self.gyro_bias = gyro_bias if gyro_bias is not None else np.zeros(3)
        
        # 初始化标志和收敛策略
        self.initialized = False
        self.convergence_phase = True
        self.convergence_time = 10.0  # 前10秒为收敛阶段
        self.start_time = None
        
        # 收敛阶段的过程噪声（更大，允许更快调整）
        self.Q_convergence = np.eye(9)
        self.Q_convergence[0:3, 0:3] *= 0.05   # 位置过程噪声（稍大）
        self.Q_convergence[3:6, 3:6] *= 0.5    # 速度过程噪声（更大）
        self.Q_convergence[6:9, 6:9] *= 0.1    # 姿态过程噪声（显著增大，允许快速调整）
        
        print(f"EKF初始化:")
        print(f"  外参矩阵: 已设置 {'(预估计)' if imu_to_world_rotation is not None else '(单位矩阵)'}")
        print(f"  陀螺仪偏置: [{self.gyro_bias[0]:.4f}, {self.gyro_bias[1]:.4f}, {self.gyro_bias[2]:.4f}] rad/s")
        print(f"  收敛策略: 前{self.convergence_time}秒使用增大的过程噪声促进收敛")
    
    def set_gravity_vector(self, gravity_magnitude=None, latitude=None):
        """设置重力向量，考虑地理位置和实际测量"""
        
        # 1. 确定重力大小
        if gravity_magnitude is not None:
            g_mag = gravity_magnitude
            print(f"使用测量重力大小: {g_mag:.3f} m/s²")
        elif latitude is not None:
            # 考虑地理位置的重力变化
            lat_rad = np.radians(latitude)
            # WGS84重力模型简化版
            g_mag = 9.780318 * (1 + 5.3024e-3 * np.sin(lat_rad)**2 - 5.8e-6 * np.sin(2*lat_rad)**2)
            print(f"基于纬度 {latitude:.3f}° 计算重力: {g_mag:.3f} m/s²")
        else:
            g_mag = 9.81  # 标准重力
            print(f"使用标准重力: {g_mag:.3f} m/s²")
        
        # 2. 设置重力向量 (ENU坐标系：East-North-Up，重力向下)
        self.gravity = np.array([0, 0, -g_mag])  # ENU中重力向下为负Z
        
        print(f"重力向量设置为: [{self.gravity[0]:.3f}, {self.gravity[1]:.3f}, {self.gravity[2]:.3f}] m/s² (ENU坐标系)")
        
        return g_mag
    
    def initialize_state(self, gps_pos, gps_vel, imu_accel, gps_positions=None, gps_times=None, initial_attitude=None):
        """改进的状态初始化"""
        # 1. 位置：直接使用第一帧GPS位置
        self.state[0:3] = gps_pos
        
        # 2. 速度：使用GPS速度或差分计算的速度
        initial_vel = np.zeros(3)
        
        if gps_vel is not None and not np.any(np.isnan(gps_vel)) and np.linalg.norm(gps_vel[:2]) > 0.2:
            initial_vel = gps_vel
            print(f"使用GPS速度初始化: [{gps_vel[0]:.2f}, {gps_vel[1]:.2f}, {gps_vel[2]:.2f}] m/s")
        else:
            # 尝试从GPS轨迹差分计算初始速度
            if gps_positions is not None and gps_times is not None and len(gps_positions) >= 2:
                pos_diff = gps_positions[1] - gps_positions[0]
                time_diff = gps_times[1] - gps_times[0] if len(gps_times) > 1 else 0.1
                
                if time_diff > 0 and np.linalg.norm(pos_diff[:2]) > 0.5:
                    initial_vel[:2] = pos_diff[:2] / time_diff
                    print(f"从GPS轨迹差分计算初始速度: [{initial_vel[0]:.2f}, {initial_vel[1]:.2f}, {initial_vel[2]:.2f}] m/s")
                else:
                    print("GPS轨迹差分无效，使用零速度初始化")
            else:
                print("GPS数据不足，使用零速度初始化")
        
        self.state[3:6] = initial_vel
        
        # 3. 姿态初始化
        if initial_attitude is not None:
            self.state[6:9] = initial_attitude
        else:
            self.state[6:9] = self.estimate_initial_attitude_improved(
                gps_vel, imu_accel, gps_positions, gps_times
            )
            
        self.initialized = True
        self.start_time = None  # 将在第一次预测时设置
        
        print(f"EKF状态初始化完成:")
        print(f"  位置: [{self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}] m")
        print(f"  速度: [{self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f}] m/s") 
        print(f"  姿态: roll={np.degrees(self.state[6]):.1f}°, pitch={np.degrees(self.state[7]):.1f}°, yaw={np.degrees(self.state[8]):.1f}°")
    
    def estimate_initial_attitude_improved(self, gps_vel, imu_accel, gps_positions=None, gps_times=None):
        """改进的姿态初始化策略"""
        roll, pitch, yaw = 0.0, 0.0, 0.0
        
        print("开始姿态初始化...")
        
        # 1. 航向角（yaw）：优先使用GPS速度向量
        yaw_estimated = False
        
        if gps_vel is not None and not np.any(np.isnan(gps_vel)):
            speed_2d = np.linalg.norm(gps_vel[:2])
            print(f"GPS速度检查: 向量=[{gps_vel[0]:.3f}, {gps_vel[1]:.3f}], 2D大小={speed_2d:.3f}m/s")
            
            if speed_2d > 0.2:  # 降低阈值
                yaw = safe_arctan2(gps_vel[0], gps_vel[1])  # atan2(East, North)
                print(f"✓ 从GPS速度向量估计航向: {np.degrees(yaw):.1f}°")
                yaw_estimated = True
            else:
                print("GPS速度太小，尝试从轨迹估计航向")
        
        # 2. 如果GPS速度不可用，计算差分速度
        if not yaw_estimated and gps_positions is not None and gps_times is not None:
            print("尝试从GPS轨迹差分计算初始速度和航向...")
            
            if len(gps_positions) >= 3:
                # 使用前几个点计算平均速度向量
                vel_vectors = []
                for i in range(1, min(5, len(gps_positions))):  # 使用前4个间隔
                    pos_diff = gps_positions[i] - gps_positions[i-1]
                    time_diff = gps_times[i] - gps_times[i-1] if i < len(gps_times) else 0.1
                    
                    if time_diff > 0 and np.linalg.norm(pos_diff[:2]) > 1.0:  # 至少1米移动
                        vel_vec = pos_diff[:2] / time_diff  # 只取水平分量
                        vel_vectors.append(vel_vec)
                        print(f"  轨迹差分 {i}: pos_diff=[{pos_diff[0]:.2f}, {pos_diff[1]:.2f}]m, dt={time_diff:.2f}s, vel=[{vel_vec[0]:.2f}, {vel_vec[1]:.2f}]m/s")
                
                if len(vel_vectors) > 0:
                    # 平均速度向量
                    avg_vel = np.mean(vel_vectors, axis=0)
                    avg_speed = np.linalg.norm(avg_vel)
                    
                    print(f"  平均差分速度: [{avg_vel[0]:.2f}, {avg_vel[1]:.2f}]m/s, 大小={avg_speed:.2f}m/s")
                    
                    if avg_speed > 0.5:
                        yaw = safe_arctan2(avg_vel[0], avg_vel[1])
                        print(f"✓ 从GPS轨迹差分估计航向: {np.degrees(yaw):.1f}°")
                        yaw_estimated = True
        
        # 3. 最后回退到轨迹方法
        if not yaw_estimated:
            print("使用原有轨迹方法估计航向...")
            if gps_positions is not None and gps_times is not None:
                yaw = self.estimate_initial_yaw_from_gps(gps_positions, gps_times)
                if yaw != 0.0:
                    print(f"✓ 从GPS轨迹估计航向: {np.degrees(yaw):.1f}°")
                    yaw_estimated = True
        
        if not yaw_estimated:
            print("⚠ 无法估计航向角，使用0°")
        
        # 2. 俯仰角和横滚角：使用IMU加速度的粗估计
        if not np.any(np.isnan(imu_accel)):
            # 将IMU加速度转换到对齐坐标系
            accel_aligned = self.imu_to_world_rotation @ imu_accel
            
            # 假设当前加速度主要是重力（适用于相对稳定的初始时刻）
            gravity_magnitude = np.linalg.norm(accel_aligned)
            
            if gravity_magnitude > 5.0:  # 合理的加速度范围
                # 归一化
                g_unit = accel_aligned / gravity_magnitude
                
                # 计算roll和pitch（假设yaw已知）
                # 考虑机动加速度的影响，使用更保守的估计
                roll = np.arctan2(g_unit[1], g_unit[2]) * 0.5  # 减小权重
                pitch = np.arctan2(-g_unit[0], np.sqrt(g_unit[1]**2 + g_unit[2]**2)) * 0.5
                
                print(f"从IMU粗估计姿态: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
            else:
                print("IMU加速度异常，使用零姿态初始化")
        
        return np.array([roll, pitch, yaw])
        
    def estimate_initial_attitude(self, imu_accel, initial_yaw=0.0):
        """从IMU加速度和GPS轨迹估计初始姿态"""
        # 将IMU坐标系的重力对齐到世界坐标系
        accel_world = self.imu_to_world_rotation @ imu_accel
        
        # 从重力向量估计roll和pitch
        gravity_norm = np.linalg.norm(accel_world)
        if gravity_norm > 1e-6:
            # 归一化重力向量
            g_unit = accel_world / gravity_norm
            
            # 计算roll和pitch
            roll = np.arctan2(g_unit[1], g_unit[2])
            pitch = np.arctan2(-g_unit[0], np.sqrt(g_unit[1]**2 + g_unit[2]**2))
            yaw = initial_yaw  # 使用GPS轨迹估计的yaw
            
            print(f"初始姿态估计: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
            
            return np.array([roll, pitch, yaw])
        
        return np.array([0.0, 0.0, initial_yaw])
    
    def estimate_initial_yaw_from_gps(self, gps_positions, gps_times, window_size=10):
        """从GPS轨迹估计初始yaw角度"""
        if len(gps_positions) < window_size:
            print("GPS数据点不足，无法估计初始航向角")
            return 0.0
        
        # 使用初始一段轨迹计算平均运动方向
        position_diffs = []
        time_diffs = []
        
        for i in range(1, min(window_size, len(gps_positions))):
            pos_diff = gps_positions[i] - gps_positions[i-1]
            time_diff = gps_times[i] - gps_times[i-1]
            
            # 只考虑有明显运动的时刻
            if np.linalg.norm(pos_diff[:2]) > 0.5 and time_diff > 0:  # 水平移动超过0.5米
                position_diffs.append(pos_diff[:2])  # 只取x,y分量
                time_diffs.append(time_diff)
        
        if len(position_diffs) == 0:
            print("初始阶段无明显运动，yaw角设为0")
            return 0.0
        
        # 计算加权平均运动方向
        weighted_direction = np.zeros(2)
        total_weight = 0
        
        for pos_diff, time_diff in zip(position_diffs, time_diffs):
            speed = np.linalg.norm(pos_diff) / time_diff
            weight = speed * time_diff  # 权重 = 速度 * 时间间隔
            
            weighted_direction += pos_diff * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_direction /= total_weight
            
            # 计算yaw角度 (北向为0°，顺时针为正)
            # GPS坐标系: x=East, y=North
            yaw = np.arctan2(weighted_direction[0], weighted_direction[1])  # atan2(East, North)
            
            print(f"从GPS轨迹估计初始航向:")
            print(f"  运动方向: East={weighted_direction[0]:.2f}m, North={weighted_direction[1]:.2f}m")
            print(f"  初始航向角: {np.degrees(yaw):.1f}° (北向为0°)")
            
            return yaw
        
        print("无法从GPS轨迹估计航向角，设为0")
        return 0.0
        
    def predict(self, imu_accel, imu_gyro, dt, current_time=None):
        """EKF预测步骤，支持自适应过程噪声"""
        if not self.initialized or dt <= 0:
            return
        
        # 检查输入数据有效性
        if (np.any(np.isnan(imu_accel)) or np.any(np.isinf(imu_accel)) or
            np.any(np.isnan(imu_gyro)) or np.any(np.isinf(imu_gyro)) or
            np.isnan(dt) or np.isinf(dt)):
            print("警告: IMU数据或时间间隔包含NaN/Inf，跳过预测步骤")
            return
        
        # 设置开始时间和检查收敛阶段
        if self.start_time is None:
            self.start_time = current_time if current_time is not None else 0.0
            
        if current_time is not None and self.convergence_phase:
            elapsed_time = current_time - self.start_time
            if elapsed_time > self.convergence_time:
                self.convergence_phase = False
                print(f"收敛阶段完成 (经过 {elapsed_time:.1f}s)，切换到正常过程噪声")
            
        # 提取当前状态
        pos = self.state[0:3]
        vel = self.state[3:6]
        attitude = self.state[6:9]
        
        # 检查状态有效性
        if np.any(np.isnan(self.state)):
            print("警告: 状态向量包含NaN，重置为零")
            self.state = np.zeros(9)
            pos = self.state[0:3]
            vel = self.state[3:6]
            attitude = self.state[6:9]
        
        # 1. 去除陀螺仪偏置
        imu_gyro_corrected = imu_gyro - self.gyro_bias
        
        # 2. 将IMU数据转换到对齐的坐标系
        accel_aligned = self.imu_to_world_rotation @ imu_accel
        gyro_aligned = self.imu_to_world_rotation @ imu_gyro_corrected
        
        # 2. 然后应用当前姿态旋转到世界坐标系
        rot = R.from_euler('xyz', attitude)
        rot_matrix = rot.as_matrix()
        
        # 3. 将加速度从机体坐标系转换到世界坐标系，并减去重力
        accel_world = rot_matrix @ accel_aligned - self.gravity
        
        # 4. 状态预测
        new_pos = pos + vel * dt + 0.5 * accel_world * dt**2
        new_vel = vel + accel_world * dt
        new_attitude = attitude + gyro_aligned * dt
        
        # 5. 姿态角度归一化 (-π到π)
        new_attitude = self._normalize_angles(new_attitude)
        
        # 更新状态
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel  
        self.state[6:9] = new_attitude
        
        # 计算雅可比矩阵F
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt  # 位置对速度的偏导
        F[3:6, 6:9] = self._compute_rotation_jacobian(accel_aligned, attitude, dt)
        
        # 选择过程噪声矩阵（收敛阶段 vs 正常阶段）
        Q_current = self.Q_convergence if self.convergence_phase else self.Q
        
        # 协方差预测
        self.P = F @ self.P @ F.T + Q_current * dt
        
    def _normalize_angles(self, angles):
        """将角度归一化到 [-π, π] 范围"""
        normalized = angles.copy()
        for i in range(len(normalized)):
            while normalized[i] > np.pi:
                normalized[i] -= 2 * np.pi
            while normalized[i] < -np.pi:
                normalized[i] += 2 * np.pi
        return normalized
        
    def _compute_rotation_jacobian(self, accel, attitude, dt):
        """计算旋转对加速度的雅可比矩阵"""
        # 简化的雅可比矩阵计算
        # 在实际应用中，这里需要更精确的计算
        roll, pitch, yaw = attitude
        
        # 简化的雅可比矩阵
        J = np.zeros((3, 3))
        
        # 这是一个简化版本，实际应用中需要更复杂的计算
        norm_accel = np.linalg.norm(accel)
        if norm_accel > 0:
            J[0, 1] = -norm_accel * np.sin(pitch) * dt  # x对pitch的偏导
            J[1, 0] = norm_accel * np.cos(pitch) * np.sin(roll) * dt  # y对roll的偏导
            J[2, 0] = -norm_accel * np.cos(pitch) * np.cos(roll) * dt  # z对roll的偏导
            
        return J
        
    def update_gps(self, gps_pos, gps_accuracy, gps_vel=None, gps_vel_accuracy=None):
        """GPS观测更新，支持位置和速度"""
        if not self.initialized:
            return
        
        # 检查GPS位置数据有效性
        if (np.any(np.isnan(gps_pos)) or np.any(np.isinf(gps_pos)) or
            np.isnan(gps_accuracy) or np.isinf(gps_accuracy) or gps_accuracy <= 0):
            print("警告: GPS位置数据无效，跳过更新步骤")
            return
        
        # 检查是否有有效的GPS速度
        use_velocity = (gps_vel is not None and 
                       not np.any(np.isnan(gps_vel)) and 
                       not np.any(np.isinf(gps_vel)) and
                       np.linalg.norm(gps_vel[:2]) > 0.1)  # 至少0.1m/s的水平速度
        
        if use_velocity:
            # 同时观测位置和速度 (6维观测)
            H = np.zeros((6, 9))
            H[0:3, 0:3] = np.eye(3)  # 位置观测
            H[3:6, 3:6] = np.eye(3)  # 速度观测
            
            # 观测向量
            z = np.concatenate([gps_pos, gps_vel])
            
            # 观测残差
            y = z - np.concatenate([self.state[0:3], self.state[3:6]])
            
            # 观测噪声矩阵
            R = np.eye(6)
            R[0:3, 0:3] *= max(gps_accuracy**2, 0.1)  # 位置噪声
            
            # GPS速度精度估计
            if gps_vel_accuracy is not None and not np.isnan(gps_vel_accuracy):
                vel_noise = max(gps_vel_accuracy**2, 0.01)  # 至少1cm/s标准差
            else:
                # 基于速度大小估计噪声
                speed = np.linalg.norm(gps_vel[:2])
                vel_noise = max(speed * 0.1, 0.01)**2  # 10%的速度误差
            
            R[3:6, 3:6] *= vel_noise  # 速度噪声
            
        else:
            # 只观测位置 (3维观测)
            H = np.zeros((3, 9))
            H[0:3, 0:3] = np.eye(3)
            
            # 观测残差
            y = gps_pos - self.state[0:3]
            
            # 观测噪声矩阵
            R = np.eye(3) * max(gps_accuracy**2, 0.1)
        
        # 检查残差合理性
        if use_velocity:
            pos_error = np.linalg.norm(y[0:3])
            vel_error = np.linalg.norm(y[3:6])
            if pos_error > 1000 or vel_error > 50:  # 位置1km或速度50m/s
                print(f"警告: GPS残差异常 (位置:{pos_error:.1f}m, 速度:{vel_error:.1f}m/s)，跳过更新")
                return
        else:
            if np.any(np.isnan(y)) or np.linalg.norm(y) > 1000:
                print(f"警告: GPS位置残差异常 ({np.linalg.norm(y):.1f}m)，跳过更新")
                return
        
        try:
            # 卡尔曼增益
            S = H @ self.P @ H.T + R
            
            # 检查S的条件数
            if np.linalg.cond(S) > 1e12:
                print("警告: 协方差矩阵S条件数过大，跳过更新")
                return
                
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # 检查增益矩阵
            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                print("警告: 卡尔曼增益包含NaN/Inf，跳过更新")
                return
            
            # 状态更新
            self.state += K @ y
            
            # 检查更新后状态
            if np.any(np.isnan(self.state)):
                print("警告: 状态更新后包含NaN，恢复上一状态")
                self.state -= K @ y
                return
            
            # 协方差更新
            I_KH = np.eye(9) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            
            # 检查协方差矩阵
            if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
                print("警告: 协方差矩阵包含NaN/Inf，重置为单位矩阵")
                self.P = np.eye(9) * 0.1
                
        except np.linalg.LinAlgError as e:
            print(f"警告: 线性代数错误 {e}，跳过GPS更新")
        
    def get_position(self):
        """获取当前估计位置"""
        return self.state[0:3].copy()
        
    def get_velocity(self):
        """获取当前估计速度"""
        return self.state[3:6].copy()
        
    def get_attitude(self):
        """获取当前估计姿态"""
        return self.state[6:9].copy()
    
    def update_magnetometer(self, mag_field, mag_noise_std=0.1, magnetic_declination=0.0):
        """磁力计观测更新，主要用于航向角修正"""
        if not self.initialized:
            return
        
        # 检查磁力计数据有效性
        if (np.any(np.isnan(mag_field)) or np.any(np.isinf(mag_field)) or 
            np.linalg.norm(mag_field) < 1e-6):
            return
        
        try:
            # 将磁力计数据转换到对齐的坐标系
            mag_aligned = self.imu_to_world_rotation @ mag_field
            
            # 归一化磁场向量
            mag_norm = np.linalg.norm(mag_aligned)
            if mag_norm > 0:
                mag_unit = mag_aligned / mag_norm
                
                # 计算磁北航向角（忽略倾角影响的简化版本）
                mag_yaw = np.arctan2(mag_unit[0], mag_unit[1]) + np.radians(magnetic_declination)
                
                # 航向角归一化到[-π, π]
                while mag_yaw > np.pi:
                    mag_yaw -= 2 * np.pi
                while mag_yaw < -np.pi:
                    mag_yaw += 2 * np.pi
                
                # 1维观测矩阵 (只观测yaw角)
                H = np.zeros((1, 9))
                H[0, 8] = 1.0  # 观测yaw状态
                
                # 观测残差
                yaw_error = mag_yaw - self.state[8]
                # 处理角度绕圈
                while yaw_error > np.pi:
                    yaw_error -= 2 * np.pi
                while yaw_error < -np.pi:
                    yaw_error += 2 * np.pi
                
                y = np.array([yaw_error])
                
                # 观测噪声
                R = np.array([[mag_noise_std**2]])
                
                # 卡尔曼增益和状态更新
                S = H @ self.P @ H.T + R
                if np.abs(S[0, 0]) > 1e-12:  # 避免除零
                    K = self.P @ H.T / S[0, 0]
                    self.state += K.flatten() * y[0]
                    
                    # 协方差更新
                    I_KH = np.eye(9) - K @ H
                    self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
                    
                    # 归一化姿态角
                    self.state[6:9] = self._normalize_angles(self.state[6:9])
                    
        except Exception as e:
            print(f"磁力计更新错误: {e}")
    
    def update_barometer(self, pressure, temperature=20.0, pressure_noise_std=100.0, sea_level_pressure=101325.0):
        """气压计观测更新，用于高度修正"""
        if not self.initialized:
            return
        
        # 检查气压数据有效性  
        if np.isnan(pressure) or np.isinf(pressure) or pressure <= 0:
            return
            
        try:
            # 气压高度公式 (国际标准大气模型简化版)
            # h = (T0/L) * [(p0/p)^(R*L/g*M) - 1]
            # 简化版本: h ≈ 44300 * (1 - (p/p0)^0.1903)
            altitude = 44300.0 * (1.0 - (pressure / sea_level_pressure) ** 0.1903)
            
            # 1维观测矩阵 (只观测z位置)
            H = np.zeros((1, 9))
            H[0, 2] = 1.0  # 观测z状态
            
            # 观测残差
            y = np.array([altitude - self.state[2]])
            
            # 观测噪声 (考虑气压变化和温度影响)
            # 气压噪声转换为高度噪声：dh ≈ -dP * R * T / (g * M * P)
            height_noise_std = pressure_noise_std * 8.314 * (temperature + 273.15) / (9.81 * 0.029 * pressure)
            R = np.array([[height_noise_std**2]])
            
            # 卡尔曼增益和状态更新
            S = H @ self.P @ H.T + R
            if np.abs(S[0, 0]) > 1e-12:
                K = self.P @ H.T / S[0, 0]
                self.state += K.flatten() * y[0]
                
                # 协方差更新
                I_KH = np.eye(9) - K @ H
                self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
                
        except Exception as e:
            print(f"气压计更新错误: {e}")


def read_magnetometer_data(provider, downsample=1):
    """读取磁力计数据"""
    try:
        # 尝试不同的磁力计流标签
        stream_labels = ["magnetometer", "mag", "compass"]
        stream_id = None
        
        for label in stream_labels:
            try:
                stream_id = provider.get_stream_id_from_label(label)
                print(f"找到磁力计流: {label}")
                break
            except:
                continue
                
        if stream_id is None:
            print("未找到磁力计数据流")
            return None, None
            
    except Exception as e:
        print(f"无法找到磁力计流: {e}")
        return None, None

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        print("磁力计流中没有数据")
        return None, None

    timestamps = []
    magnetic_fields = []

    for idx in range(0, num_samples, downsample):
        try:
            mag_data = provider.get_magnetometer_data_by_index(stream_id, idx)
            
            # 调试：检查第一个磁力计数据的结构
            if idx == 0:
                print(f"磁力计数据类型: {type(mag_data)}")
                print(f"磁力计数据属性: {[attr for attr in dir(mag_data) if not attr.startswith('_')]}")
            
            # 提取磁力计数据 (通常是3轴磁场强度)
            mag_field = np.array([mag_data.mag_tesla[0], mag_data.mag_tesla[1], mag_data.mag_tesla[2]])
            timestamp = mag_data.capture_timestamp_ns * 1e-9
            
            magnetic_fields.append(mag_field)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"读取磁力计数据错误 (索引 {idx}): {e}")
            continue

    if len(timestamps) == 0:
        return None, None
        
    return np.array(timestamps), np.array(magnetic_fields)


def read_barometer_data(provider, downsample=1):
    """读取气压计数据"""
    try:
        # 尝试不同的气压计流标签
        stream_labels = ["barometer", "baro", "pressure"]
        stream_id = None
        
        for label in stream_labels:
            try:
                stream_id = provider.get_stream_id_from_label(label)
                print(f"找到气压计流: {label}")
                break
            except:
                continue
                
        if stream_id is None:
            print("未找到气压计数据流")
            return None, None
            
    except Exception as e:
        print(f"无法找到气压计流: {e}")
        return None, None

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        print("气压计流中没有数据")
        return None, None

    timestamps = []
    pressures = []
    temperatures = []

    for idx in range(0, num_samples, downsample):
        try:
            baro_data = provider.get_barometer_data_by_index(stream_id, idx)
            
            # 调试：检查第一个气压计数据的结构
            if idx == 0:
                print(f"气压计数据类型: {type(baro_data)}")
                print(f"气压计数据属性: {[attr for attr in dir(baro_data) if not attr.startswith('_')]}")
            
            # 提取气压和温度数据
            pressure = baro_data.pressure_pascal  # 压强 (Pa)
            temperature = getattr(baro_data, 'temperature_celsius', 20.0)  # 温度 (可能不可用)
            timestamp = baro_data.capture_timestamp_ns * 1e-9
            
            pressures.append(pressure)
            temperatures.append(temperature)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"读取气压计数据错误 (索引 {idx}): {e}")
            continue

    if len(timestamps) == 0:
        return None, None
        
    return np.array(timestamps), np.array(pressures), np.array(temperatures)


def read_imu_data(provider, downsample=1):
    """读取IMU数据"""
    try:
        # 尝试不同的IMU流标签
        stream_labels = ["imu-right", "imu-left", "imu", "accelerometer", "gyroscope"]
        stream_id = None
        
        for label in stream_labels:
            try:
                stream_id = provider.get_stream_id_from_label(label)
                print(f"找到IMU流: {label}")
                break
            except:
                continue
                
        if stream_id is None:
            raise RuntimeError("未找到IMU数据流")
            
    except Exception as e:
        raise RuntimeError(f"无法找到IMU流: {e}")

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        raise RuntimeError("IMU流中没有数据")

    timestamps = []
    accelerations = []
    angular_velocities = []

    for idx in range(0, num_samples, downsample):
        try:
            imu_data = provider.get_imu_data_by_index(stream_id, idx)
            
            # 调试：检查第一个IMU数据的结构
            if idx == 0:
                print(f"IMU数据类型: {type(imu_data)}")
                print(f"IMU数据属性: {[attr for attr in dir(imu_data) if not attr.startswith('_')]}")
            
            # 提取数据
            accel = np.array([imu_data.accel_msec2[0], imu_data.accel_msec2[1], imu_data.accel_msec2[2]])
            gyro = np.array([imu_data.gyro_radsec[0], imu_data.gyro_radsec[1], imu_data.gyro_radsec[2]])
            timestamp = imu_data.capture_timestamp_ns * 1e-9
            
            accelerations.append(accel)
            angular_velocities.append(gyro)
            timestamps.append(timestamp)
            
        except Exception as e:
            print(f"读取IMU数据错误 (索引 {idx}): {e}")
            continue

    return (np.array(timestamps), 
            np.array(accelerations), 
            np.array(angular_velocities))


def read_gps_data(provider, downsample=1):
    """读取GPS数据，包括速度信息"""
    try:
        stream_id = provider.get_stream_id_from_label("gps")
    except Exception:
        raise RuntimeError("未找到GPS流")

    num_samples = provider.get_num_data(stream_id)
    if num_samples == 0:
        raise RuntimeError("GPS流中没有数据")

    timestamps = []
    positions = []
    accuracies = []
    velocities = []

    for idx in range(0, num_samples, downsample):
        try:
            gps_data = provider.get_gps_data_by_index(stream_id, idx)
            
            # 调试：检查GPS数据结构（仅第一个数据点）
            if idx == 0:
                print(f"GPS数据字段检查:")
                all_attrs = [attr for attr in dir(gps_data) if not attr.startswith('_')]
                print(f"  可用属性: {all_attrs}")
                
                # 检查速度相关字段
                velocity_fields = ['speed', 'velocity', 'bearing', 'course', 'vx', 'vy', 'vz']
                for field in velocity_fields:
                    if hasattr(gps_data, field):
                        value = getattr(gps_data, field)
                        print(f"    {field}: {value}")
            
            lat = gps_data.latitude
            lon = gps_data.longitude  
            alt = gps_data.altitude
            acc = gps_data.accuracy
            timestamp = gps_data.capture_timestamp_ns * 1e-9

            positions.append([lat, lon, alt])
            accuracies.append(acc)
            timestamps.append(timestamp)
            
            # 尝试获取速度信息
            vel = np.zeros(3)  # 默认值
            
            if hasattr(gps_data, 'speed') and hasattr(gps_data, 'bearing'):
                # 从速度和方位角计算速度向量
                speed = gps_data.speed  # m/s
                bearing = gps_data.bearing  # 通常是度数
                
                if not (np.isnan(speed) or np.isnan(bearing)) and speed > 0.1:
                    bearing_rad = np.radians(bearing)
                    vel[0] = speed * np.sin(bearing_rad)  # East
                    vel[1] = speed * np.cos(bearing_rad)  # North
                    vel[2] = 0.0  # 垂直速度通常不可用
                    
                    if idx == 0:  # 调试第一个速度
                        print(f"    速度计算: speed={speed:.2f}m/s, bearing={bearing:.1f}°")
                        print(f"    速度向量: East={vel[0]:.2f}, North={vel[1]:.2f}")
                        
            elif hasattr(gps_data, 'speed') and not hasattr(gps_data, 'bearing'):
                # 只有速度大小，没有方位角，需要从位置差分估计方向
                speed = gps_data.speed
                
                if idx == 0:
                    print(f"    检测到GPS速度: {speed:.2f}m/s (无方位角，稍后从轨迹估计方向)")
                
                if not np.isnan(speed) and speed > 0.1:
                    # 暂时存储速度大小，方向将在后处理中从GPS轨迹估计
                    # 这里先存储为标量形式，后面会转换
                    vel = [speed, 0, 0]  # 临时存储格式：[速度大小, 0, 0]
                    
            elif hasattr(gps_data, 'vx') and hasattr(gps_data, 'vy'):
                # 直接的速度分量
                vel[0] = getattr(gps_data, 'vx', 0.0)
                vel[1] = getattr(gps_data, 'vy', 0.0) 
                vel[2] = getattr(gps_data, 'vz', 0.0)
                
                if idx == 0 and (abs(vel[0]) > 0.1 or abs(vel[1]) > 0.1):
                    print(f"    直接速度: vx={vel[0]:.2f}, vy={vel[1]:.2f}, vz={vel[2]:.2f}")
            
            velocities.append(vel)
            
            # 调试：检查是否有有效速度
            if idx == 0:
                speed_mag = np.linalg.norm(vel[:2])
                print(f"    最终速度向量: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s")
                print(f"    2D速度大小: {speed_mag:.2f} m/s")
            
        except Exception as e:
            print(f"读取GPS数据错误 (索引 {idx}): {e}")
            continue

    velocities = np.array(velocities)
    
    # 后处理：如果速度向量只有大小没有方向，从GPS轨迹估计方向
    if len(velocities) > 0 and velocities.shape[1] == 3:
        needs_direction_estimation = False
        
        for i in range(len(velocities)):
            # 检查是否是临时存储格式 [速度大小, 0, 0]
            if velocities[i, 0] > 0.1 and velocities[i, 1] == 0.0 and velocities[i, 2] == 0.0:
                needs_direction_estimation = True
                break
        
        if needs_direction_estimation:
            print("后处理：从GPS轨迹估计速度方向...")
            positions_array = np.array(positions)
            timestamps_array = np.array(timestamps)
            
            # 计算GPS轨迹的局部运动方向
            for i in range(len(velocities)):
                if velocities[i, 0] > 0.1 and velocities[i, 1] == 0.0:  # 需要估计方向
                    speed_magnitude = velocities[i, 0]
                    
                    # 使用前后几个点的平均运动方向
                    direction_vector = estimate_velocity_direction(i, positions_array, timestamps_array)
                    
                    if direction_vector is not None:
                        # 归一化方向向量并乘以速度大小
                        direction_norm = np.linalg.norm(direction_vector)
                        if direction_norm > 0:
                            unit_direction = direction_vector / direction_norm
                            velocities[i, 0] = speed_magnitude * unit_direction[0]  # East
                            velocities[i, 1] = speed_magnitude * unit_direction[1]  # North
                            velocities[i, 2] = 0.0  # 垂直分量
                            
                            if i == 0:  # 调试第一个估计结果
                                print(f"    第一个速度向量估计结果: East={velocities[i, 0]:.2f}, North={velocities[i, 1]:.2f} m/s")
    
    return (np.array(timestamps),
            np.array(positions),
            np.array(accuracies),
            velocities)


def estimate_velocity_direction(current_idx, positions, timestamps, window_size=5):
    """根据GPS轨迹估计当前点的速度方向"""
    if len(positions) < 2:
        return None
    
    # 定义前后窗口范围
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(positions), current_idx + window_size + 1)
    
    # 收集有效的位置差分向量
    direction_vectors = []
    weights = []
    
    for i in range(start_idx + 1, end_idx):
        if i < len(positions):
            # 计算位置差分
            pos_diff = np.array(positions[i][:2]) - np.array(positions[i-1][:2])  # 只取经纬度
            
            # 转换经纬度差分为米制
            # 简化计算：使用中点纬度的Earth radius修正
            lat_mid = (positions[i][0] + positions[i-1][0]) / 2
            R_earth = 6378137.0
            lat_mid_rad = np.radians(lat_mid)
            
            # 转换为东向和北向距离（米）
            east_m = pos_diff[1] * R_earth * np.cos(lat_mid_rad) * np.pi / 180
            north_m = pos_diff[0] * R_earth * np.pi / 180
            
            direction_m = np.array([east_m, north_m])
            distance = np.linalg.norm(direction_m)
            
            # 只考虑有明显移动的点
            if distance > 0.5:  # 至少0.5米移动
                # 计算权重：距离当前点越近权重越大
                distance_to_current = abs(i - current_idx)
                weight = 1.0 / (1.0 + distance_to_current)
                
                direction_vectors.append(direction_m)
                weights.append(weight)
    
    if len(direction_vectors) == 0:
        return None
    
    # 计算加权平均方向
    direction_vectors = np.array(direction_vectors)
    weights = np.array(weights)
    
    weighted_direction = np.sum(direction_vectors * weights[:, np.newaxis], axis=0)
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        return weighted_direction / total_weight
    
    return None


def gps_to_local_coords(gps_positions, origin_gps):
    """将GPS坐标转换为局部坐标系 (米)"""
    R_earth = 6378137.0  # 地球半径
    
    origin_lat, origin_lon, origin_alt = origin_gps
    lat_rad = np.radians(gps_positions[:, 0])
    lon_rad = np.radians(gps_positions[:, 1])
    alt = gps_positions[:, 2]
    
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    
    # 转换为局部坐标
    x = (lon_rad - origin_lon_rad) * R_earth * np.cos(origin_lat_rad)
    y = (lat_rad - origin_lat_rad) * R_earth
    z = alt - origin_alt
    
    return np.column_stack([x, y, z])


def validate_timestamps(timestamps, data_name):
    """验证时间戳的有效性"""
    if len(timestamps) == 0:
        raise ValueError(f"{data_name}: 时间戳数组为空")
    
    # 检查单调性
    time_diffs = np.diff(timestamps)
    non_monotonic = np.sum(time_diffs <= 0)
    if non_monotonic > 0:
        print(f"警告: {data_name} 有 {non_monotonic} 个非单调时间戳")
        
    # 检查时间间隔
    if len(time_diffs) > 0:
        avg_dt = np.mean(time_diffs[time_diffs > 0])
        max_dt = np.max(time_diffs)
        min_dt = np.min(time_diffs[time_diffs > 0])
        
        print(f"{data_name} 时间戳统计:")
        print(f"  平均间隔: {avg_dt*1000:.1f}ms")
        print(f"  最大间隔: {max_dt*1000:.1f}ms") 
        print(f"  最小间隔: {min_dt*1000:.1f}ms")
        
        # 检查异常间隔
        large_gaps = time_diffs > avg_dt * 5
        if np.sum(large_gaps) > 0:
            print(f"  发现 {np.sum(large_gaps)} 个大间隔 (>5倍平均值)")
    
    return timestamps

def interpolate_data_advanced(timestamps_target, timestamps_source, data_source, method='linear'):
    """改进的数据插值函数，支持多种插值方法"""
    from scipy import interpolate
    
    # 验证输入
    if len(timestamps_source) != len(data_source):
        raise ValueError(f"时间戳长度 {len(timestamps_source)} 与数据长度 {len(data_source)} 不匹配")
    
    # 处理单维数据
    if len(data_source.shape) == 1:
        if method == 'linear':
            return np.interp(timestamps_target, timestamps_source, data_source)
        elif method == 'cubic':
            # 三次样条插值（需要至少4个点）
            if len(timestamps_source) >= 4:
                f = interpolate.interp1d(timestamps_source, data_source, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')
                return f(timestamps_target)
            else:
                return np.interp(timestamps_target, timestamps_source, data_source)
    
    # 处理多维数据
    result = np.zeros((len(timestamps_target), data_source.shape[1]))
    for i in range(data_source.shape[1]):
        if method == 'linear':
            result[:, i] = np.interp(timestamps_target, timestamps_source, data_source[:, i])
        elif method == 'cubic' and len(timestamps_source) >= 4:
            f = interpolate.interp1d(timestamps_source, data_source[:, i], kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
            result[:, i] = f(timestamps_target)
        else:
            result[:, i] = np.interp(timestamps_target, timestamps_source, data_source[:, i])
    
    return result

def synchronize_sensors(gps_times, gps_data, imu_times, imu_data, 
                       gps_delay=0.0, imu_delay=0.0):
    """改进的传感器时间同步"""
    
    print("开始传感器时间同步...")
    
    # 验证时间戳
    gps_times_clean = validate_timestamps(gps_times, "GPS")
    imu_times_clean = validate_timestamps(imu_times, "IMU")
    
    # 应用传感器延迟补偿
    gps_times_compensated = gps_times_clean + gps_delay
    imu_times_compensated = imu_times_clean + imu_delay
    
    if gps_delay != 0:
        print(f"应用GPS延迟补偿: {gps_delay*1000:.1f}ms")
    if imu_delay != 0:
        print(f"应用IMU延迟补偿: {imu_delay*1000:.1f}ms")
    
    # 找到公共时间范围（加入缓冲区）
    start_time = max(gps_times_compensated[0], imu_times_compensated[0])
    end_time = min(gps_times_compensated[-1], imu_times_compensated[-1])
    
    # 检查时间重叠
    if start_time >= end_time:
        raise ValueError(f"GPS和IMU数据没有时间重叠: GPS[{gps_times_compensated[0]:.1f}, {gps_times_compensated[-1]:.1f}], IMU[{imu_times_compensated[0]:.1f}, {imu_times_compensated[-1]:.1f}]")
    
    # 创建融合时间戳（以IMU频率为主，但去除异常值）
    imu_dts = np.diff(imu_times_compensated)
    valid_dts = imu_dts[(imu_dts > 0) & (imu_dts < np.percentile(imu_dts, 95))]
    dt_imu = np.median(valid_dts)
    
    # 确保时间范围合理
    max_duration = min(3600, end_time - start_time)  # 最多1小时
    if end_time - start_time > max_duration:
        end_time = start_time + max_duration
        print(f"限制融合时间范围为 {max_duration/60:.1f} 分钟")
    
    fusion_times = np.arange(start_time, end_time, dt_imu)
    
    print(f"同步结果:")
    print(f"  融合时间范围: {start_time:.1f} - {end_time:.1f} 秒 (时长: {end_time-start_time:.1f}秒)")
    print(f"  IMU采样间隔: {dt_imu*1000:.1f} ms")
    print(f"  融合数据点数: {len(fusion_times)}")
    
    return fusion_times, gps_times_compensated, imu_times_compensated


def clean_sensor_data(times, data, data_name):
    """清理传感器数据中的NaN和异常值"""
    print(f"清理{data_name}数据...")
    
    original_count = len(times)
    if original_count == 0:
        return times, data
    
    # 检查时间戳
    time_valid = ~(np.isnan(times) | np.isinf(times))
    
    # 检查数据
    if len(data.shape) == 1:
        data_valid = ~(np.isnan(data) | np.isinf(data))
    else:
        data_valid = ~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))
    
    # 综合有效性
    valid_mask = time_valid & data_valid
    valid_count = np.sum(valid_mask)
    
    if valid_count == 0:
        raise ValueError(f"{data_name}: 所有数据都无效(NaN/Inf)")
    
    # 过滤数据
    clean_times = times[valid_mask]
    clean_data = data[valid_mask]
    
    removed_count = original_count - valid_count
    if removed_count > 0:
        print(f"  移除了 {removed_count}/{original_count} 个无效数据点 ({removed_count/original_count*100:.1f}%)")
        
        # 检查时间间隔是否合理
        if len(clean_times) > 1:
            time_gaps = np.diff(clean_times)
            large_gaps = time_gaps > np.median(time_gaps) * 5
            if np.sum(large_gaps) > 0:
                print(f"  警告: 发现 {np.sum(large_gaps)} 个大时间间隔 (可能影响插值)")
    
    return clean_times, clean_data

def validate_and_fix_rotation_matrix(R):
    """验证和修复旋转矩阵"""
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        print("警告: 旋转矩阵包含NaN/Inf，使用单位矩阵")
        return np.eye(3)
    
    # 检查是否接近正交矩阵
    should_be_identity = R @ R.T
    identity_error = np.max(np.abs(should_be_identity - np.eye(3)))
    
    if identity_error > 0.1:
        print(f"警告: 旋转矩阵不正交 (误差: {identity_error:.3f})，进行修正")
        # 使用SVD修正
        U, _, Vt = np.linalg.svd(R)
        R_corrected = U @ Vt
        
        # 确保行列式为+1 (右手坐标系)
        if np.linalg.det(R_corrected) < 0:
            U[:, -1] *= -1
            R_corrected = U @ Vt
            
        return R_corrected
    
    return R

def safe_arctan2(y, x):
    """安全的atan2，处理NaN输入"""
    if np.any(np.isnan(y)) or np.any(np.isnan(x)):
        print("警告: arctan2输入包含NaN")
        return 0.0
    
    if np.abs(x) < 1e-12 and np.abs(y) < 1e-12:
        return 0.0
    
    return np.arctan2(y, x)

def estimate_global_extrinsics(gps_times, gps_positions, imu_times, imu_accels, imu_gyros):
    """使用全部数据估计IMU到世界坐标系的外参"""
    print("开始全局外参估计...")
    
    # 数据清理
    gps_times_clean, gps_positions_clean = clean_sensor_data(gps_times, gps_positions, "GPS")
    imu_times_clean, imu_accels_clean = clean_sensor_data(imu_times, imu_accels, "IMU加速度")
    imu_times_clean, imu_gyros_clean = clean_sensor_data(imu_times, imu_gyros, "IMU陀螺仪")
    
    # 1. 估计陀螺仪偏置（假设整体平均角速度接近0）
    gyro_bias = np.nanmean(imu_gyros_clean, axis=0)
    
    # 检查偏置是否有效
    if np.any(np.isnan(gyro_bias)) or np.any(np.abs(gyro_bias) > 1.0):  # 偏置不应超过1 rad/s
        print("警告: 陀螺仪偏置异常，使用零偏置")
        gyro_bias = np.zeros(3)
    
    print(f"陀螺仪偏置估计: [{gyro_bias[0]:.4f}, {gyro_bias[1]:.4f}, {gyro_bias[2]:.4f}] rad/s")
    
    # 去除偏置
    imu_gyros_corrected = imu_gyros_clean - gyro_bias
    
    # 2. 使用GPS轨迹估计初始航向角
    initial_yaw = 0.0
    if len(gps_positions_clean) > 10:
        # 使用前20%的轨迹计算平均运动方向
        end_idx = min(len(gps_positions_clean), max(10, len(gps_positions_clean) // 5))
        
        total_displacement = np.zeros(2)
        total_weight = 0
        
        for i in range(1, end_idx):
            pos_diff = gps_positions_clean[i] - gps_positions_clean[i-1]
            
            # 检查位置差是否有效
            if np.any(np.isnan(pos_diff)):
                continue
                
            weight = np.linalg.norm(pos_diff[:2])  # 距离作为权重
            
            if weight > 0.5:  # 只考虑明显运动
                total_displacement += pos_diff[:2] * weight
                total_weight += weight
        
        if total_weight > 0 and not np.any(np.isnan(total_displacement)):
            avg_direction = total_displacement / total_weight
            initial_yaw = safe_arctan2(avg_direction[0], avg_direction[1])  # atan2(East, North)
            print(f"初始航向估计: {np.degrees(initial_yaw):.1f}° (从GPS轨迹)")
        else:
            print("无法从GPS轨迹估计航向角，使用0°")
    
    # 3. 使用低通滤波估计重力方向
    # 对加速度数据进行低通滤波
    imu_to_world_rotation = np.eye(3)  # 默认值
    
    if len(imu_accels_clean) > 100:
        try:
            # 简单的移动平均滤波
            window_size = min(50, len(imu_accels_clean) // 10)
            filtered_accels = []
            
            for i in range(len(imu_accels_clean)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(imu_accels_clean), i + window_size // 2 + 1)
                window_data = imu_accels_clean[start_idx:end_idx]
                
                # 使用nanmean处理可能的NaN
                filtered_accel = np.nanmean(window_data, axis=0)
                if not np.any(np.isnan(filtered_accel)):
                    filtered_accels.append(filtered_accel)
            
            if len(filtered_accels) == 0:
                print("警告: 滤波后无有效加速度数据，使用单位矩阵")
            else:
                filtered_accels = np.array(filtered_accels)
                
                # 使用中段数据估计重力方向（避免开始和结束的不稳定）
                mid_start = len(filtered_accels) // 4
                mid_end = 3 * len(filtered_accels) // 4
                mid_accels = filtered_accels[mid_start:mid_end]
                
                mean_accel = np.nanmean(mid_accels, axis=0)
                
                if np.any(np.isnan(mean_accel)):
                    print("警告: 平均加速度包含NaN，使用单位矩阵")
                else:
                    gravity_magnitude = np.linalg.norm(mean_accel)
                    
                    print(f"重力向量估计: [{mean_accel[0]:.2f}, {mean_accel[1]:.2f}, {mean_accel[2]:.2f}] m/s²")
                    print(f"重力模长: {gravity_magnitude:.2f} m/s²")
                    
                    if 5.0 < gravity_magnitude < 15.0:  # 合理的重力范围
                        gravity_imu = mean_accel / gravity_magnitude
                        
                        # 理想的重力方向 (ENU坐标系中向下为负Z)
                        gravity_world = np.array([0, 0, -1])  # ENU: 重力向下
                        
                        # 计算旋转矩阵来对齐重力向量
                        v = np.cross(gravity_imu, gravity_world)
                        s = np.linalg.norm(v)
                        c = np.dot(gravity_imu, gravity_world)
                        
                        if s > 1e-6:  # 避免除零
                            vx = np.array([[0, -v[2], v[1]],
                                          [v[2], 0, -v[0]],
                                          [-v[1], v[0], 0]])
                            
                            imu_to_world_rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                            
                            # 验证和修复旋转矩阵
                            imu_to_world_rotation = validate_and_fix_rotation_matrix(imu_to_world_rotation)
                        else:
                            print("重力向量已对齐，使用单位矩阵")
                    else:
                        print(f"警告: 重力向量异常 ({gravity_magnitude:.2f} m/s²)，使用单位矩阵")
        except Exception as e:
            print(f"重力向量估计出错: {e}，使用单位矩阵")
            imu_to_world_rotation = np.eye(3)
    else:
        print("警告: IMU数据不足，使用单位矩阵")
    
    # 4. 进一步优化外参（可选：基于GPS-IMU一致性）
    # 提取实际测量的重力大小
    measured_gravity = None
    if len(imu_accels_clean) > 100:
        try:
            window_size = min(50, len(imu_accels_clean) // 10)
            filtered_accels = []
            for i in range(len(imu_accels_clean)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(imu_accels_clean), i + window_size // 2 + 1)
                window_data = imu_accels_clean[start_idx:end_idx]
                filtered_accel = np.nanmean(window_data, axis=0)
                if not np.any(np.isnan(filtered_accel)):
                    filtered_accels.append(filtered_accel)
            
            if len(filtered_accels) > 0:
                filtered_accels = np.array(filtered_accels)
                mid_start = len(filtered_accels) // 4
                mid_end = 3 * len(filtered_accels) // 4
                mid_accels = filtered_accels[mid_start:mid_end]
                mean_accel = np.nanmean(mid_accels, axis=0)
                
                if not np.any(np.isnan(mean_accel)):
                    measured_gravity = np.linalg.norm(mean_accel)
        except:
            pass
    
    print("外参估计完成:")
    print(f"  旋转矩阵条件数: {np.linalg.cond(imu_to_world_rotation):.2f}")
    print(f"  是否正交: {np.allclose(imu_to_world_rotation @ imu_to_world_rotation.T, np.eye(3))}")
    if measured_gravity is not None:
        print(f"  测量重力大小: {measured_gravity:.3f} m/s²")
    
    return imu_to_world_rotation, gyro_bias, initial_yaw, measured_gravity


def run_ekf_fusion(gps_times, gps_positions, gps_accuracies, gps_velocities, 
                   imu_times, imu_accels, imu_gyros, 
                   mag_times=None, mag_fields=None, 
                   baro_times=None, baro_pressures=None, baro_temps=None):
    """运行EKF融合算法"""
    
    print("开始EKF融合算法...")
    
    # 步骤0: 数据清理
    print("清理输入数据...")
    
    # 确保GPS数据的时间戳、位置、精度和速度长度一致
    min_gps_len = min(len(gps_times), len(gps_positions), len(gps_accuracies), len(gps_velocities))
    gps_times_trim = gps_times[:min_gps_len]
    gps_positions_trim = gps_positions[:min_gps_len]  
    gps_accuracies_trim = gps_accuracies[:min_gps_len]
    gps_velocities_trim = gps_velocities[:min_gps_len]
    
    print(f"GPS数据长度调整: times={len(gps_times)}→{min_gps_len}, pos={len(gps_positions)}→{min_gps_len}, acc={len(gps_accuracies)}→{min_gps_len}, vel={len(gps_velocities)}→{min_gps_len}")
    
    # 清理GPS数据（保持一致性）
    gps_times_clean, gps_positions_clean = clean_sensor_data(gps_times_trim, gps_positions_trim, "GPS位置")
    
    # 使用相同的有效掩码清理精度和速度数据
    valid_gps_mask = ~(np.isnan(gps_times_trim) | np.isinf(gps_times_trim) | 
                       np.isnan(gps_positions_trim).any(axis=1) | np.isinf(gps_positions_trim).any(axis=1))
    
    gps_accuracies_clean = gps_accuracies_trim[valid_gps_mask]
    gps_velocities_clean = gps_velocities_trim[valid_gps_mask]
    
    print(f"GPS数据清理后: {len(gps_times_clean)} 个有效数据点")
    
    # 步骤1: 全局外参估计
    imu_to_world_rotation, gyro_bias, initial_yaw, measured_gravity = estimate_global_extrinsics(
        gps_times_clean, gps_positions_clean, imu_times, imu_accels, imu_gyros
    )
    
    # 步骤2: 传感器时间同步（使用清理后的数据）
    fusion_times, gps_times_sync, imu_times_sync = synchronize_sensors(
        gps_times_clean, gps_positions_clean, imu_times, imu_accels
    )
    
    # 步骤3: 数据插值
    print("开始数据插值...")
    print(f"调试信息:")
    print(f"  fusion_times长度: {len(fusion_times)}")
    print(f"  gps_times_sync长度: {len(gps_times_sync)}")  
    print(f"  gps_positions_clean长度: {len(gps_positions_clean)}")
    print(f"  gps_accuracies_clean长度: {len(gps_accuracies_clean)}")
    print(f"  imu_times_sync长度: {len(imu_times_sync)}")
    print(f"  imu_accels长度: {len(imu_accels)}")
    
    # 确保使用相同长度的清理后数据
    try:
        interp_gps_pos = interpolate_data_advanced(fusion_times, gps_times_sync, gps_positions_clean, method='linear')
        interp_gps_acc = interpolate_data_advanced(fusion_times, gps_times_sync, gps_accuracies_clean, method='linear')
        
        # IMU数据使用三次样条插值以保持平滑性
        interp_imu_accel = interpolate_data_advanced(fusion_times, imu_times_sync, imu_accels, method='cubic')
        interp_imu_gyro = interpolate_data_advanced(fusion_times, imu_times_sync, imu_gyros, method='cubic')
        
        print(f"插值后数据长度:")
        print(f"  interp_gps_pos: {interp_gps_pos.shape}")
        print(f"  interp_gps_acc: {interp_gps_acc.shape}")
        print(f"  interp_imu_accel: {interp_imu_accel.shape}")
        print(f"  interp_imu_gyro: {interp_imu_gyro.shape}")
        
        # 检查数据长度一致性
        expected_len = len(fusion_times)
        if (len(interp_gps_pos) != expected_len or len(interp_gps_acc) != expected_len or
            len(interp_imu_accel) != expected_len or len(interp_imu_gyro) != expected_len):
            
            print(f"警告: 插值后数据长度不一致，预期长度: {expected_len}")
            print("截断到最短长度...")
            
            min_len = min(len(interp_gps_pos), len(interp_gps_acc), 
                         len(interp_imu_accel), len(interp_imu_gyro), expected_len)
            
            fusion_times = fusion_times[:min_len]
            interp_gps_pos = interp_gps_pos[:min_len]
            interp_gps_acc = interp_gps_acc[:min_len]
            interp_imu_accel = interp_imu_accel[:min_len]
            interp_imu_gyro = interp_imu_gyro[:min_len]
            
            print(f"调整后统一长度: {min_len}")
        
    except Exception as e:
        print(f"插值过程出错: {e}")
        print("尝试使用原始数据进行插值...")
        
        # 回退到使用原始数据
        try:
            interp_gps_pos = interpolate_data_advanced(fusion_times, gps_times, gps_positions, method='linear')
            interp_gps_acc = interpolate_data_advanced(fusion_times, gps_times, gps_accuracies, method='linear')
            interp_imu_accel = interpolate_data_advanced(fusion_times, imu_times, imu_accels, method='cubic')
            interp_imu_gyro = interpolate_data_advanced(fusion_times, imu_times, imu_gyros, method='cubic')
        except Exception as e2:
            print(f"原始数据插值也失败: {e2}")
            print("使用线性插值作为最后回退...")
            
            interp_gps_pos = interpolate_data_advanced(fusion_times, gps_times, gps_positions, method='linear')
            interp_gps_acc = interpolate_data_advanced(fusion_times, gps_times, gps_accuracies, method='linear') 
            interp_imu_accel = interpolate_data_advanced(fusion_times, imu_times, imu_accels, method='linear')
            interp_imu_gyro = interpolate_data_advanced(fusion_times, imu_times, imu_gyros, method='linear')
    
    # 步骤4: 初始化EKF（使用预估计的外参）
    ekf = EKF(imu_to_world_rotation=imu_to_world_rotation, gyro_bias=gyro_bias)
    
    # 设置重力向量
    if len(gps_positions_clean) > 0:
        origin_lat = gps_positions_clean[0][0]  # GPS纬度
        actual_gravity = ekf.set_gravity_vector(gravity_magnitude=measured_gravity, latitude=origin_lat)
    else:
        actual_gravity = ekf.set_gravity_vector(gravity_magnitude=measured_gravity)
    
    # 初始化状态（使用改进的初始化方法）
    # 计算初始GPS速度（如果插值数据可用）
    initial_gps_vel = None
    if len(gps_velocities_clean) > 0:
        # 插值GPS速度到fusion_times
        try:
            interp_gps_vel = interpolate_data_advanced(fusion_times, gps_times_clean, gps_velocities_clean, method='linear')
            initial_gps_vel = interp_gps_vel[0]
        except:
            initial_gps_vel = gps_velocities_clean[0]  # 使用原始第一个速度
    
    ekf.initialize_state(interp_gps_pos[0], initial_gps_vel, interp_imu_accel[0], 
                        interp_gps_pos, fusion_times)
    
    # 存储结果
    ekf_positions = [ekf.get_position()]
    ekf_velocities = [ekf.get_velocity()]  
    ekf_attitudes = [ekf.get_attitude()]
    
    # 插值GPS速度数据
    print("插值GPS速度数据...")
    interp_gps_vel = None
    if len(gps_velocities_clean) > 0:
        try:
            interp_gps_vel = interpolate_data_advanced(fusion_times, gps_times_clean, gps_velocities_clean, method='linear')
            print(f"GPS速度插值成功: {interp_gps_vel.shape}")
        except Exception as e:
            print(f"GPS速度插值失败: {e}，EKF将仅使用位置观测")
            interp_gps_vel = None
    else:
        print("无GPS速度数据，EKF将仅使用位置观测")
        
    # 插值磁力计数据
    interp_mag_fields = None
    if mag_times is not None and mag_fields is not None:
        try:
            print("插值磁力计数据...")
            interp_mag_fields = interpolate_data_advanced(fusion_times, mag_times, mag_fields, method='linear')
            print(f"磁力计插值成功: {interp_mag_fields.shape}")
        except Exception as e:
            print(f"磁力计插值失败: {e}")
            interp_mag_fields = None
    
    # 插值气压计数据
    interp_baro_pressures = None
    interp_baro_temps = None
    if baro_times is not None and baro_pressures is not None:
        try:
            print("插值气压计数据...")
            interp_baro_pressures = interpolate_data_advanced(fusion_times, baro_times, baro_pressures, method='linear')
            if baro_temps is not None:
                interp_baro_temps = interpolate_data_advanced(fusion_times, baro_times, baro_temps, method='linear')
            print(f"气压计插值成功: {interp_baro_pressures.shape}")
        except Exception as e:
            print(f"气压计插值失败: {e}")
            interp_baro_pressures = None
            interp_baro_temps = None

    # 步骤5: 主融合循环
    print("开始EKF融合...")
    for i in range(1, len(fusion_times)):
        t = fusion_times[i]
        dt = t - fusion_times[i-1]
        
        # EKF预测步骤 (使用IMU数据)
        ekf.predict(interp_imu_accel[i], interp_imu_gyro[i], dt, current_time=t)
        
        # EKF更新步骤 (使用GPS数据)
        # 降低GPS更新频率以模拟实际情况
        if i % 10 == 0:  # 每10个IMU周期更新一次GPS
            # 使用GPS位置和速度（如果可用）进行更新
            gps_vel_for_update = interp_gps_vel[i] if interp_gps_vel is not None else None
            ekf.update_gps(interp_gps_pos[i], interp_gps_acc[i], gps_vel_for_update)
        
        # 磁力计更新 (如果可用，频率较低)
        if interp_mag_fields is not None and i % 20 == 0:  # 每20个IMU周期更新一次磁力计
            ekf.update_magnetometer(interp_mag_fields[i], mag_noise_std=0.1)
        
        # 气压计更新 (如果可用，频率较低)
        if interp_baro_pressures is not None and i % 15 == 0:  # 每15个IMU周期更新一次气压计
            temp = interp_baro_temps[i] if interp_baro_temps is not None else 20.0
            ekf.update_barometer(interp_baro_pressures[i], temperature=temp, pressure_noise_std=100.0)
        
        # 保存结果
        ekf_positions.append(ekf.get_position())
        ekf_velocities.append(ekf.get_velocity())
        ekf_attitudes.append(ekf.get_attitude())
    
    print(f"EKF融合完成，处理了 {len(ekf_positions)} 个数据点")
    
    return (fusion_times,
            np.array(ekf_positions),
            np.array(ekf_velocities), 
            np.array(ekf_attitudes),
            interp_gps_pos)


def unwrap_angles(angles_deg):
    """解决角度绕圈问题，使用角度增量累积的方法"""
    angles = np.array(angles_deg)
    if len(angles) == 0:
        return angles
        
    unwrapped = np.zeros_like(angles)
    unwrapped[0] = angles[0]
    
    for i in range(1, len(angles)):
        # 计算角度增量
        delta = angles[i] - angles[i-1]
        
        # 将角度增量归一化到 [-180, 180] 范围
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
            
        # 累积角度增量
        unwrapped[i] = unwrapped[i-1] + delta
    
    return unwrapped


def plot_comparison(fusion_times, ekf_positions, gps_positions, ekf_attitudes, out_prefix, ekf_velocities=None, gps_velocities=None):
    """绘制EKF前后对比图"""
    
    # 1. 位置轨迹对比图
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('GPS/IMU EKF Fusion Results', fontsize=16)
    
    # 2D轨迹对比
    axes[0, 0].plot(gps_positions[:, 0], gps_positions[:, 1], 'r-', alpha=0.7, label='GPS Only')
    axes[0, 0].plot(ekf_positions[:, 0], ekf_positions[:, 1], 'b-', linewidth=2, label='EKF Fused')
    axes[0, 0].scatter(gps_positions[0, 0], gps_positions[0, 1], c='green', s=100, marker='s', label='Start')
    axes[0, 0].scatter(gps_positions[-1, 0], gps_positions[-1, 1], c='red', s=100, marker='^', label='End')
    axes[0, 0].set_xlabel('East (m)')
    axes[0, 0].set_ylabel('North (m)')
    axes[0, 0].set_title('2D Trajectory Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # 3D轨迹对比
    ax_3d = fig.add_subplot(3, 3, 2, projection='3d')
    ax_3d.plot(gps_positions[:, 0], gps_positions[:, 1], gps_positions[:, 2], 
               'r-', alpha=0.7, label='GPS Only')
    ax_3d.plot(ekf_positions[:, 0], ekf_positions[:, 1], ekf_positions[:, 2], 
               'b-', linewidth=2, label='EKF Fused')
    
    # 添加XY平面投影
    z_min = min(min(gps_positions[:, 2]), min(ekf_positions[:, 2]))
    z_range = max(max(gps_positions[:, 2]), max(ekf_positions[:, 2])) - z_min
    projection_z = z_min - 0.1 * z_range if z_range > 0 else z_min - 1
    
    # 绘制XY平面投影轨迹
    ax_3d.plot(gps_positions[:, 0], gps_positions[:, 1], projection_z, 
               'r-', alpha=0.4, linewidth=1, label='GPS Projection')
    ax_3d.plot(ekf_positions[:, 0], ekf_positions[:, 1], projection_z, 
               'b-', alpha=0.4, linewidth=1, label='EKF Projection')
    
    ax_3d.set_xlabel('East (m)')
    ax_3d.set_ylabel('North (m)')
    ax_3d.set_zlabel('Up (m)')
    ax_3d.set_title('3D Trajectory with XY Projection')
    ax_3d.legend()
    
    # 姿态角时间序列（解决角度绕圈问题）
    roll_deg_unwrapped = unwrap_angles(np.degrees(ekf_attitudes[:, 0]))
    pitch_deg_unwrapped = unwrap_angles(np.degrees(ekf_attitudes[:, 1]))
    yaw_deg_unwrapped = unwrap_angles(np.degrees(ekf_attitudes[:, 2]))
    
    axes[0, 2].plot(fusion_times, roll_deg_unwrapped, 'r-', label='Roll')
    axes[0, 2].plot(fusion_times, pitch_deg_unwrapped, 'g-', label='Pitch') 
    axes[0, 2].plot(fusion_times, yaw_deg_unwrapped, 'b-', label='Yaw')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Angle (degrees, unwrapped)')
    axes[0, 2].set_title('EKF Attitude Estimates (Roll/Pitch/Yaw)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 位置误差时间序列
    position_error = np.linalg.norm(ekf_positions - gps_positions, axis=1)
    axes[1, 0].plot(fusion_times, position_error, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position Error (m)')
    axes[1, 0].set_title('EKF vs GPS Position Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 各轴位置对比
    axes[1, 1].plot(fusion_times, gps_positions[:, 0], 'r--', alpha=0.7, label='GPS East')
    axes[1, 1].plot(fusion_times, ekf_positions[:, 0], 'b-', label='EKF East')
    axes[1, 1].plot(fusion_times, gps_positions[:, 1], 'r:', alpha=0.7, label='GPS North')  
    axes[1, 1].plot(fusion_times, ekf_positions[:, 1], 'g-', label='EKF North')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position (m)')
    axes[1, 1].set_title('Position Components vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 航向角对比 (从GPS速度计算 vs EKF估计)
    gps_headings = []
    
    # 直接用前后两个点差值计算航向
    for i in range(1, len(gps_positions)):
        pos_diff = gps_positions[i] - gps_positions[i-1]
        heading = np.arctan2(pos_diff[0], pos_diff[1])  # atan2(East, North)
        gps_headings.append(np.degrees(heading))
    
    # 第一个点使用第二个点的航向
    if len(gps_headings) > 0:
        gps_headings = [gps_headings[0]] + gps_headings
    else:
        gps_headings = [0] * len(gps_positions)
    
    # 解决角度绕圈问题
    gps_headings_unwrapped = unwrap_angles(gps_headings)
    ekf_yaw_degrees = np.degrees(ekf_attitudes[:, 2])
    ekf_yaw_unwrapped = unwrap_angles(ekf_yaw_degrees)
    
    # 绘制展开后的航向角
    axes[1, 2].plot(fusion_times, gps_headings_unwrapped, 'r--', alpha=0.7, linewidth=1, 
                   label='GPS Heading (from motion)')
    axes[1, 2].plot(fusion_times, ekf_yaw_unwrapped, 'b-', linewidth=2, label='EKF Yaw')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Heading (degrees, unwrapped)')
    axes[1, 2].set_title('Heading Comparison (Continuous)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 添加角度统计信息
    if len(gps_headings_unwrapped) > 0 and len(ekf_yaw_unwrapped) > 0:
        angle_diff = ekf_yaw_unwrapped - gps_headings_unwrapped
        mean_angle_error = np.mean(np.abs(angle_diff))
        print(f"航向角误差统计:")
        print(f"  平均航向角误差: {mean_angle_error:.2f}°")
        print(f"  航向角误差标准差: {np.std(angle_diff):.2f}°")
        print(f"  最大航向角误差: {np.max(np.abs(angle_diff)):.2f}°")
    
    # 速度对比（第三行）
    if ekf_velocities is not None:
        # EKF速度大小时间序列
        ekf_speed = np.linalg.norm(ekf_velocities[:, :2], axis=1)  # 2D速度大小
        axes[2, 0].plot(fusion_times, ekf_speed, 'b-', linewidth=2, label='EKF Speed')
        
        if gps_velocities is not None:
            gps_speed = np.linalg.norm(gps_velocities[:, :2], axis=1)
            axes[2, 0].plot(fusion_times, gps_speed, 'r--', alpha=0.7, label='GPS Speed')
            axes[2, 0].set_title('Speed Comparison')
        else:
            axes[2, 0].set_title('EKF Speed Estimates')
            
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Speed (m/s)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 速度分量对比
        axes[2, 1].plot(fusion_times, ekf_velocities[:, 0], 'b-', linewidth=2, label='EKF Vx (East)')
        axes[2, 1].plot(fusion_times, ekf_velocities[:, 1], 'g-', linewidth=2, label='EKF Vy (North)')
        
        if gps_velocities is not None:
            axes[2, 1].plot(fusion_times, gps_velocities[:, 0], 'r--', alpha=0.7, label='GPS Vx')
            axes[2, 1].plot(fusion_times, gps_velocities[:, 1], 'orange', linestyle='--', alpha=0.7, label='GPS Vy')
            axes[2, 1].set_title('Velocity Components Comparison')
        else:
            axes[2, 1].set_title('EKF Velocity Components')
            
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Velocity (m/s)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 第三个图：如果有GPS速度则显示误差，否则显示其他信息
        if gps_velocities is not None:
            # 速度误差
            vel_error = np.linalg.norm(ekf_velocities[:, :2] - gps_velocities[:, :2], axis=1)
            axes[2, 2].plot(fusion_times, vel_error, 'purple', linewidth=1)
            axes[2, 2].set_xlabel('Time (s)')
            axes[2, 2].set_ylabel('Velocity Error (m/s)')
            axes[2, 2].set_title('EKF vs GPS Velocity Difference')
            axes[2, 2].grid(True, alpha=0.3)
            
            print(f"速度误差统计:")
            print(f"  平均速度误差: {np.mean(vel_error):.3f} m/s")
            print(f"  速度误差标准差: {np.std(vel_error):.3f} m/s")
            print(f"  最大速度误差: {np.max(vel_error):.3f} m/s")
        else:
            # 显示EKF速度变化率（加速度）
            ekf_accel = np.zeros_like(ekf_speed)
            if len(fusion_times) > 1:
                dt = np.diff(fusion_times)
                speed_diff = np.diff(ekf_speed)
                ekf_accel[1:] = speed_diff / dt
                
            axes[2, 2].plot(fusion_times, ekf_accel, 'purple', linewidth=1)
            axes[2, 2].set_xlabel('Time (s)')
            axes[2, 2].set_ylabel('Speed Change Rate (m/s²)')
            axes[2, 2].set_title('EKF Speed Acceleration')
            axes[2, 2].grid(True, alpha=0.3)
    else:
        # 如果没有EKF速度数据，显示提示信息
        for col in range(3):
            axes[2, col].text(0.5, 0.5, 'EKF velocity\ndata unavailable', 
                             transform=axes[2, col].transAxes, ha='center', va='center', fontsize=12)
            axes[2, col].set_title('Velocity Data Unavailable')
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_ekf_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 统计信息
    mean_error = np.mean(position_error)
    std_error = np.std(position_error)
    max_error = np.max(position_error)
    
    print(f"\nEKF融合结果统计:")
    print(f"  平均位置误差: {mean_error:.2f} m")
    print(f"  位置误差标准差: {std_error:.2f} m") 
    print(f"  最大位置误差: {max_error:.2f} m")
    print(f"  轨迹总长度 (GPS): {calculate_trajectory_length(gps_positions):.1f} m")
    print(f"  轨迹总长度 (EKF): {calculate_trajectory_length(ekf_positions):.1f} m")


def calculate_trajectory_length(positions):
    """计算轨迹总长度"""
    if len(positions) < 2:
        return 0.0
    
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.sum(distances)


def main():
    parser = argparse.ArgumentParser(description="Project Aria GPS/IMU EKF Fusion for Odometry")
    parser.add_argument("vrs_path", type=str, help="Path to the Project Aria VRS file")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor for data (default: 1)")
    parser.add_argument("--out-prefix", type=str, default="ekf_fusion", 
                        help="Output file prefix (default: ekf_fusion)")
    
    args = parser.parse_args()
    
    print(f"正在处理 VRS 文件: {args.vrs_path}")
    
    # 打开VRS文件
    try:
        provider = data_provider.create_vrs_data_provider(args.vrs_path)
    except Exception as e:
        print(f"无法打开VRS文件: {e}")
        sys.exit(1)
    
    # 读取GPS数据
    try:
        print("读取GPS数据...")
        gps_times, gps_positions, gps_accuracies, gps_velocities = read_gps_data(provider, args.downsample)
        print(f"GPS数据点数: {len(gps_times)}")
        print(f"GPS速度数据范围: {np.min(np.linalg.norm(gps_velocities, axis=1)):.2f} - {np.max(np.linalg.norm(gps_velocities, axis=1)):.2f} m/s")
    except Exception as e:
        print(f"读取GPS数据失败: {e}")
        sys.exit(1)
    
    # 读取IMU数据  
    try:
        print("读取IMU数据...")
        imu_times, imu_accels, imu_gyros = read_imu_data(provider, args.downsample)
        print(f"IMU数据点数: {len(imu_times)}")
    except Exception as e:
        print(f"读取IMU数据失败: {e}")
        sys.exit(1)
    
    # 读取磁力计数据（可选）
    mag_times, mag_fields = None, None
    try:
        print("尝试读取磁力计数据...")
        mag_times, mag_fields = read_magnetometer_data(provider, args.downsample)
        if mag_times is not None:
            print(f"磁力计数据点数: {len(mag_times)}")
        else:
            print("磁力计数据不可用")
    except Exception as e:
        print(f"读取磁力计数据失败: {e}")
    
    # 读取气压计数据（可选）
    baro_times, baro_pressures, baro_temps = None, None, None
    try:
        print("尝试读取气压计数据...")
        baro_times, baro_pressures, baro_temps = read_barometer_data(provider, args.downsample)
        if baro_times is not None:
            print(f"气压计数据点数: {len(baro_times)}")
        else:
            print("气压计数据不可用")
    except Exception as e:
        print(f"读取气压计数据失败: {e}")
    
    # 转换GPS到局部坐标系
    print("转换GPS坐标...")
    origin_gps = gps_positions[0]  # 使用第一个GPS点作为原点
    gps_local = gps_to_local_coords(gps_positions, origin_gps)
    
    print(f"坐标原点: 纬度={origin_gps[0]:.6f}°, 经度={origin_gps[1]:.6f}°, 海拔={origin_gps[2]:.1f}m")
    
    # 转换GPS速度到局部坐标系
    print("转换GPS速度...")
    # GPS速度通常已经在ENU坐标系，只需要检查和转换
    gps_velocities_local = gps_velocities.copy()  # 速度不需要坐标转换
    
    # 运行EKF融合
    print("运行EKF融合...")
    try:
        (fusion_times, ekf_positions, ekf_velocities, 
         ekf_attitudes, interp_gps) = run_ekf_fusion(
            gps_times, gps_local, gps_accuracies, gps_velocities_local,
            imu_times, imu_accels, imu_gyros,
            mag_times, mag_fields, 
            baro_times, baro_pressures, baro_temps
        )
        print(f"EKF融合完成，生成 {len(fusion_times)} 个数据点")
    except Exception as e:
        print(f"EKF融合失败: {e}")
        sys.exit(1)
    
    # 绘制对比图
    print("生成对比图...")
    # 准备GPS速度数据用于对比（如果可用）
    gps_vel_for_plot = None
    if 'interp_gps_vel' in locals() and interp_gps_vel is not None:
        gps_vel_for_plot = interp_gps_vel
    
    plot_comparison(fusion_times, ekf_positions, interp_gps, ekf_attitudes, args.out_prefix, 
                   ekf_velocities=ekf_velocities, gps_velocities=gps_vel_for_plot)
    
    print(f"\n完成！生成文件:")
    print(f"  {args.out_prefix}_ekf_comparison.png")
    print("\n提示: EKF融合结合了GPS的绝对位置和IMU的高频运动信息，")
    print("      可以提供更平滑、更高频率的轨迹估计。")


if __name__ == "__main__":
    main()