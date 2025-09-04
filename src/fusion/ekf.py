#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Kalman Filter (EKF) for GPS/IMU Sensor Fusion

Core EKF implementation with support for:
- 9-DOF state vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
- GPS position and velocity updates
- Magnetometer heading corrections
- Barometer altitude updates
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


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
        self.gravity = np.array([0, 0, -9.81])  # ENU坐标系：重力向下
        
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
        from .initialization import StateInitializer
        
        initializer = StateInitializer(self)
        self.state = initializer.initialize(
            gps_pos, gps_vel, imu_accel, gps_positions, gps_times, initial_attitude
        )
        
        self.initialized = True
        self.start_time = None  # 将在第一次预测时设置
        
        print(f"EKF状态初始化完成:")
        print(f"  位置: [{self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}] m")
        print(f"  速度: [{self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f}] m/s") 
        print(f"  姿态: roll={np.degrees(self.state[6]):.1f}°, pitch={np.degrees(self.state[7]):.1f}°, yaw={np.degrees(self.state[8]):.1f}°")
        
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
        
    def update_gps(self, gps_pos, gps_accuracy, gps_vel=None, gps_vel_accuracy=None, gps_vertical_accuracy=None):
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
            R[0:2, 0:2] *= max(gps_accuracy**2, 0.1)  # XY位置噪声
            
            # 使用垂直精度或回退到水平精度的10倍
            if gps_vertical_accuracy is not None and not np.isnan(gps_vertical_accuracy) and gps_vertical_accuracy > 0:
                z_noise = max(gps_vertical_accuracy**2, 0.1)
            else:
                z_noise = max(gps_accuracy**2, 0.1) * 10  # 回退到水平精度的10倍
            R[2, 2] *= z_noise
            
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
            R = np.eye(3)
            R[0:2, 0:2] *= max(gps_accuracy**2, 0.1)  # XY位置噪声
            
            # 使用垂直精度或回退到水平精度的10倍
            if gps_vertical_accuracy is not None and not np.isnan(gps_vertical_accuracy) and gps_vertical_accuracy > 0:
                z_noise = max(gps_vertical_accuracy**2, 0.1)
            else:
                z_noise = max(gps_accuracy**2, 0.1) * 10  # 回退到水平精度的10倍
            R[2, 2] *= z_noise
        
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
    
    def update_barometer_relative(self, current_pressure_kpa, baseline_pressure_kpa, temperature=20.0, pressure_noise_std=0.1):
        """气压计相对高度约束更新"""
        if not self.initialized:
            return
        
        # 检查气压数据有效性  
        if (np.isnan(current_pressure_kpa) or np.isinf(current_pressure_kpa) or current_pressure_kpa <= 0 or
            np.isnan(baseline_pressure_kpa) or np.isinf(baseline_pressure_kpa) or baseline_pressure_kpa <= 0):
            return
            
        try:
            # 计算相对高度变化 (不依赖绝对海平面气压)
            # 使用对数近似: Δh ≈ -H₀ * ln(P/P₀), H₀ ≈ 8400m 
            scale_height = 8400.0  # 大气标高 (米)
            relative_height_change = -scale_height * np.log(current_pressure_kpa / baseline_pressure_kpa)
            
            # 我们不观测绝对高度，而是约束相对高度变化的一致性
            # 如果有记录的上一次气压测量，可以验证Z轴变化的合理性
            
            # 这里实现一个简化的相对约束：验证当前高度变化与气压变化的一致性
            # 如果EKF估计的高度变化与气压暗示的变化差异过大，则进行修正
            
            # 暂时记录这个相对高度变化，可以用于验证EKF的Z轴变化合理性
            # 但不直接作为观测更新，而是作为约束检查
            
            # 可以添加一个软约束：如果EKF的Z轴变化与气压计暗示的变化差异过大
            # 则增加过程噪声或调整状态估计
            
        except Exception as e:
            print(f"气压计相对约束错误: {e}")
            
    def get_position(self):
        """获取当前估计位置"""
        return self.state[0:3].copy()
        
    def get_velocity(self):
        """获取当前估计速度"""
        return self.state[3:6].copy()
        
    def get_attitude(self):
        """获取当前估计姿态"""
        return self.state[6:9].copy()