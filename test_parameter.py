"""
Configuration parameters for marvel testing.

This file defines various parameters for:
- Model loading and testing configuration
- Simulation parameters
- Heading and navigation settings
- Map representation and resolution
- Sensor and utility thresholds
- Network and graph parameters
- GPU usage configuration

"""
import os

TEST_SET = 'maps_test'
LOAD_FOLDER_NAME = 'MARVEL'
load_path = f'load_model/{LOAD_FOLDER_NAME}'
gifs_path = f'results/gifs/{LOAD_FOLDER_NAME}'
LOAD_MODEL = True  
SAVE_IMG_GAP = 50
SAVE_GIFS = True
GREEDY = True
NUM_RUN = 1
NUM_TEST = 100

# Sim parameters
USE_CONTINUOUS_SIM = True
NUM_SIM_STEPS = 6
MAX_EPISODE_STEP = 128
VELOCITY = 1
YAW_RATE = 35 # in degrees

# Heading parameters
NUM_ANGLES_BIN = 36
NUM_HEADING_CANDIDATES = 3

# Map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# Map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# Sensor and utility range
MIN_UTILITY = 1

# Updating map range w.r.t the robot
UPDATING_MAP_SIZE = 15 * NODE_RESOLUTION

# Testing parameters
NUM_META_AGENT = 10
INITIAL_EXPLORED_RATE = 0.90

# Network parameters
NODE_INPUT_DIM = 6
EMBEDDING_DIM = 128

# Graph parameters
NUM_NODE_NEIGHBORS = 5
K_SIZE = NUM_NODE_NEIGHBORS**2   # the number of neighboring nodes

# GPU usage
USE_GPU = False  
NUM_GPU = 0

# Inference-only overlap coordination: off, fov, frontier, or full
OVERLAP_COORDINATION_MODE = os.environ.get('MARVEL_COORD_MODE', 'off').lower()
OVERLAP_COORDINATION_TOP_K = int(os.environ.get('MARVEL_COORD_TOP_K', '12'))
OVERLAP_COORDINATION_OVERLAP_WEIGHT = float(os.environ.get('MARVEL_COORD_OVERLAP_WEIGHT', '2.0'))
OVERLAP_COORDINATION_FRONTIER_WEIGHT = float(os.environ.get('MARVEL_COORD_FRONTIER_WEIGHT', '1.0'))
OVERLAP_COORDINATION_FRONTIER_SOFTNESS = float(os.environ.get('MARVEL_COORD_FRONTIER_SOFTNESS', '1.0'))
OVERLAP_COORDINATION_SWITCH_MARGIN = float(os.environ.get('MARVEL_COORD_SWITCH_MARGIN', '0.05'))
OVERLAP_COORDINATION_HYSTERESIS_BONUS = float(os.environ.get('MARVEL_COORD_HYSTERESIS_BONUS', '0.05'))
OVERLAP_COORDINATION_OVERLAP_TRIGGER = float(os.environ.get('MARVEL_COORD_OVERLAP_TRIGGER', '0.10'))
OVERLAP_COORDINATION_MAX_POLICY_DROP = float(os.environ.get('MARVEL_COORD_MAX_POLICY_DROP', 'inf'))
OVERLAP_COORDINATION_MAX_DETOUR_RATIO = float(os.environ.get('MARVEL_COORD_MAX_DETOUR_RATIO', 'inf'))
OVERLAP_COORDINATION_SAME_WAYPOINT_ONLY = os.environ.get('MARVEL_COORD_SAME_WAYPOINT_ONLY', '0') == '1'
OVERLAP_COORDINATION_TABU_WAYPOINT_STEPS = int(os.environ.get('MARVEL_COORD_TABU_WAYPOINT_STEPS', '0'))
OVERLAP_COORDINATION_DETOUR_BUDGET = float(os.environ.get('MARVEL_COORD_DETOUR_BUDGET', 'inf'))
OVERLAP_COORDINATION_STALL_GAIN = float(os.environ.get('MARVEL_COORD_STALL_GAIN', '0.002'))
OVERLAP_COORDINATION_STALL_STEPS = int(os.environ.get('MARVEL_COORD_STALL_STEPS', '12'))
OVERLAP_COORDINATION_MIN_FRONTIER_RATIO = float(os.environ.get('MARVEL_COORD_MIN_FRONTIER_RATIO', '0.0'))
# 拥挤态分离项：机群间距 < separation_radius 时，奖励候选落点远离已承诺队友（0 关闭）
OVERLAP_COORDINATION_SEPARATION_WEIGHT = float(os.environ.get('MARVEL_COORD_SEPARATION_WEIGHT', '0.0'))
OVERLAP_COORDINATION_SEPARATION_RADIUS = float(os.environ.get('MARVEL_COORD_SEPARATION_RADIUS', '0.0'))
OVERLAP_COORDINATION_SEPARATION_REDUCE = os.environ.get('MARVEL_COORD_SEPARATION_REDUCE', 'mean').lower()
OVERLAP_COORDINATION_SEPARATION_HORIZON = float(os.environ.get('MARVEL_COORD_SEPARATION_HORIZON', 'inf'))
