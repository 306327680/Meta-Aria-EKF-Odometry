#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities Package

Provides utility functions for:
- Coordinate transformations
- Angle handling
- Data processing
- Calibration
"""

from .coordinates import gps_to_local_coords, estimate_velocity_direction
from .angles import safe_arctan2, unwrap_angles
from .data_processing import *
from .calibration import estimate_global_extrinsics

__all__ = [
    'gps_to_local_coords',
    'estimate_velocity_direction', 
    'safe_arctan2',
    'unwrap_angles',
    'validate_timestamps',
    'interpolate_data_advanced',
    'synchronize_sensors',
    'clean_sensor_data',
    'validate_and_fix_rotation_matrix',
    'estimate_global_extrinsics'
]