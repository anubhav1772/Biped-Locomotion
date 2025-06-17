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
        default='stand'
    )
    args = parser.parse_args()

    controller = BipedMotionController()

    motion_dispatch = {
        'stand': controller.stand,
    }

    motion_fn = motion_dispatch.get(args.motion, controller.stand)
    motion_fn()

if __name__ == "__main__":
    main()
