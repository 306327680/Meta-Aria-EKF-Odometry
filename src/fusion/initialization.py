#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EKF State Initialization Module

Handles intelligent initialization of EKF state vector from sensor data.
"""

import numpy as np
from ..utils.angles import safe_arctan2


class StateInitializer:
    """EKF状态初始化器"""
    
    def __init__(self, ekf_instance):
        self.ekf = ekf_instance
    
    def initialize(self, gps_pos, gps_vel, imu_accel, gps_positions=None, gps_times=None, initial_attitude=None):
        """改进的状态初始化"""
        state = np.zeros(9)
        
        # 1. 位置：直接使用第一帧GPS位置
        state[0:3] = gps_pos
        
        # 2. 速度：使用GPS速度或差分计算的速度
        initial_vel = self._initialize_velocity(gps_vel, gps_positions, gps_times)
        state[3:6] = initial_vel
        
        # 3. 姿态初始化
        if initial_attitude is not None:
            state[6:9] = initial_attitude
        else:
            state[6:9] = self._estimate_initial_attitude_improved(
                gps_vel, imu_accel, gps_positions, gps_times
            )
            
        return state
    
    def _initialize_velocity(self, gps_vel, gps_positions, gps_times):
        """初始化速度状态"""
        initial_vel = np.zeros(3)
        
        if gps_vel is not None and not np.any(np.isnan(gps_vel)) and np.linalg.norm(gps_vel[:2]) > 0.2:
            initial_vel = gps_vel
            print(f"使用GPS速度初始化: [{gps_vel[0]:.2f}, {gps_vel[1]:.2f}, {gps_vel[2]:.2f}] m/s")
        else:
            # 尝试从GPS轨迹差分计算初始速度（使用稳定的方法）
            if gps_positions is not None and gps_times is not None and len(gps_positions) >= 3:
                # 使用多个点计算平均速度，避免单点异常影响
                valid_velocities = []
                
                for i in range(1, min(5, len(gps_positions))):  # 使用前4个间隔
                    pos_diff = gps_positions[i] - gps_positions[i-1]
                    time_diff = gps_times[i] - gps_times[i-1] if i < len(gps_times) else 0.1
                    
                    # 检查是否为合理的速度（避免异常跳变）
                    if (time_diff > 0 and 
                        np.linalg.norm(pos_diff[:2]) > 0.5 and  # 水平移动至少0.5米
                        np.linalg.norm(pos_diff[:2]) < 50.0 and  # 水平移动不超过50米（避免异常）
                        abs(pos_diff[2]) < 20.0):  # 垂直变化不超过20米（避免GPS高度跳变）
                        
                        vel = pos_diff / time_diff
                        valid_velocities.append(vel)
                        print(f"  有效差分速度 {i}: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s")
                
                if len(valid_velocities) > 0:
                    # 计算平均速度（更稳定）
                    initial_vel = np.mean(valid_velocities, axis=0)
                    print(f"从GPS轨迹稳定差分计算初始速度: [{initial_vel[0]:.2f}, {initial_vel[1]:.2f}, {initial_vel[2]:.2f}] m/s")
                else:
                    print("GPS轨迹差分数据异常，使用零速度初始化")
            else:
                print("GPS数据不足，使用零速度初始化")
        
        return initial_vel
    
    def _estimate_initial_attitude_improved(self, gps_vel, imu_accel, gps_positions=None, gps_times=None):
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
                yaw = self._estimate_initial_yaw_from_gps(gps_positions, gps_times)
                if yaw != 0.0:
                    print(f"✓ 从GPS轨迹估计航向: {np.degrees(yaw):.1f}°")
                    yaw_estimated = True
        
        if not yaw_estimated:
            print("⚠ 无法估计航向角，使用0°")
        
        # 2. 俯仰角和横滚角：使用IMU加速度的粗估计
        if not np.any(np.isnan(imu_accel)):
            # 将IMU加速度转换到对齐坐标系
            accel_aligned = self.ekf.imu_to_world_rotation @ imu_accel
            
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
    
    def _estimate_initial_yaw_from_gps(self, gps_positions, gps_times, window_size=10):
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