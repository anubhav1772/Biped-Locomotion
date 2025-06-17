import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.biped import Biped
from motion_planning.walking import PreviewControl

import matplotlib.pyplot as plt

class BipedMotionController:
    def __init__(self):
        self.biped = Biped()
        self.CoM_height = 0.45
        self.targetRPY = [0.0, 0.0, 0.0]
        self.targetPosL = [0.0, 0.065, -self.CoM_height]
        self.targetPosR = [0.0, -0.065, -self.CoM_height]

    def _update_incline(self):
        incline = self.biped.env.get_incline()
        self.biped.env.reset_incline()
        self.targetRPY[1] = incline

    def stand(self):
        self.biped.initialize_position(init_time=0.2)
        while True:
            self._update_incline()
            self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
            self.biped.env.step()

def main():
    parser = argparse.ArgumentParser(
        description="Execute a motion behavior for the robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--motion',
        help='Type of motion behavior to execute.',
        type=str,
        choices=['stand'],
        default='walk'
    )
    args = parser.parse_args()

    controller = BipedMotionController()

    motion_dispatch = {
        'stand': controller.stand,
    }

    motion_fn = motion_dispatch.get(args.motion, controller.stand)
    motion_fn()

    left_leg_joint_values = np.array(controller.biped.left_leg_joints_value_logs)
    right_leg_joint_values = np.array(controller.biped.right_leg_joints_value_logs)

    plt.figure(figsize=(12, 8))  # Adjust size as needed

    # Plot 1
    plt.subplot(2, 2, 1)
    plt.plot(CoM_x_list, label='CoM_x')
    plt.plot(ZMP_x_list, label='ZMP_x')
    plt.xlabel('Time Step')
    plt.ylabel('X Position')
    plt.title('CoM_x and ZMP_x over Time')
    plt.legend()

    # Plot 2
    plt.subplot(2, 2, 2)
    plt.plot(CoM_y_list, label='CoM_y')
    plt.plot(ZMP_y_list, label='ZMP_y')
    plt.xlabel('Time Step')
    plt.ylabel('Y Position')
    plt.title('CoM_y and ZMP_y over Time')
    plt.legend()

    # Plot 3
    plt.subplot(2, 2, 3)
    for i in range(6):
        plt.plot(left_leg_joint_values[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Value')
    plt.title('Left Leg Joint Positions over Time')
    plt.legend()

    # Plot 4
    plt.subplot(2, 2, 4)
    for i in range(6):
        plt.plot(right_leg_joint_values[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Value')
    plt.title('Right Leg Joint Positions over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
