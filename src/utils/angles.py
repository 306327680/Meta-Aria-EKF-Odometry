#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Angle Handling Utilities

Provides functions for safe angle operations and unwrapping.
"""

import numpy as np


def safe_arctan2(y, x):
    """安全的atan2，处理NaN输入"""
    if np.any(np.isnan(y)) or np.any(np.isnan(x)):
        print("警告: arctan2输入包含NaN")
        return 0.0
    
    if np.abs(x) < 1e-12 and np.abs(y) < 1e-12:
        return 0.0
    
    return np.arctan2(y, x)


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