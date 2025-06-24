import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.biped import Biped
from motion_planning.walking import PreviewControl

import matplotlib.pyplot as plt

CoM_x_list = []
CoM_y_list = []
ZMP_x_list = []
ZMP_y_list = []

class BipedMotionController:
    def __init__(self):
        self.biped = Biped()
        self.CoM_height = 0.45
        self.targetRPY = [0.0, 0.0, 0.0]
        self.targetPosL = [0.0, 0.065, -self.CoM_height]
        self.targetPosR = [0.0, -0.065, -self.CoM_height]

    def _update_incline(self):
        # read terrain slope from the environment
        incline = self.biped.env.get_incline()
        self.biped.env.reset_incline()
        # Updates the pitch angle of the foot
        self.targetRPY[1] = incline

    def stand(self):
        self.biped.initialize_position(init_time=0.2)
        while True:
            self._update_incline()
            self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
            self.biped.env.step()
    
    def walk(self):
        """
        Executes a dynamic walking cycle for the biped robot using preview control.
        """
        self.biped.initialize_position(init_time=0.2)
        # ZMP-based preview controller responsible for generating smooth CoM and foot trajectories
        pre = PreviewControl(dt=1. / 240., Tsup_time=0.3, Tdl_time=0.1, previewStepNum=190)

        # Logs all CoM positions over time
        CoM_trajectory = np.empty((0, 3), float)
        trjR_log = np.empty((0, 3), float)
        trjL_log = np.empty((0, 3), float)

        # Starting support point (initial ZMP target) - assumed under the left leg (y = +0.065)
        supPoint = np.array([0., 0.065])

        count = 0
        while count < 25:
            # 1. Inclination Handling
            self._update_incline()

            # 2. Preview Control Trajectory Generation
            # Both foot placement and ZMP are set to supPoint
            stepHeight = self.biped.get_step_height()
            CoM_trj, footTrjL, footTrjR = pre.footPrintAndCoM_trajectoryGenerator(
                inputTargetZMP=supPoint,
                inputFootPrint=supPoint,
                stepHeight=stepHeight
            )

            # Log all trajectories
            CoM_trajectory = np.vstack((CoM_trajectory, CoM_trj))
            trjR_log = np.vstack((trjR_log, footTrjR))
            trjL_log = np.vstack((trjL_log, footTrjL))

            # 3. Apply Control at Each Time Step
            for j in range(len(CoM_trj)):
                # Store CoM
                CoM_x_list.append(CoM_trj[j][0])
                CoM_y_list.append(CoM_trj[j][1])

                # Computes leg targets relative to the CoM
                self.targetPosR = footTrjR[j] - CoM_trj[j]
                self.targetPosL = footTrjL[j] - CoM_trj[j]

                self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
                # To simulate the next timestep
                self.biped.env.step()

            # 4. Update Support Leg & Stride
            # Moves the ZMP target forward by the stride length (x-direction)
            # Switches leg (y value flips sign: +0.065 -> -0.065 and vice versa)
            supPoint[0] += self.biped.get_stride()
            supPoint[1] = -supPoint[1]

            count += 1

        # Get desired ZMP from preview controller logs
        ZMP_x_list.extend(pre.px_ref_log)
        ZMP_y_list.extend(pre.py_ref_log)

def main():
    parser = argparse.ArgumentParser(
        description="Execute a motion behavior for the robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--motion',
        help='Type of motion behavior to execute.',
        type=str,
        choices=['stand', 'walk'],
        default='stand'
    )
    args = parser.parse_args()

    controller = BipedMotionController()

    motion_dispatch = {
        'stand': controller.stand,
        'walk': controller.walk,
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
