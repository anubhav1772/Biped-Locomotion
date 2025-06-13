import pybullet as p
import pybullet_data
import numpy as np
import time

from envs.base_robot import Robot
from utils.transformUtils import TransformUtils as tf
import scipy.linalg as la

class Biped(Robot):
    def __init__(
        self,
        start_pos=[0, 0, 0.55],
        start_orn=[0, 0, 0],
        com_pos=np.array([0.0, 0.0, -0.02]),
        max_force=9.0,
        control_mode=p.POSITION_CONTROL,
        urdf_path='urdf/biped_flatfoot_robot.urdf'
    ):
        super().__init__(
            urdf_path=urdf_path,
            start_pos=start_pos,
            start_orn=start_orn,
            max_force=max_force,
            control_mode=control_mode
        )

        self._lambda = 1.0
        self.l1 = 0.18
        self.l2 = 0.18
        self.leg_dof = 6

        self.right_foot_offset = np.array([0, -0.065, -0.175]) - com_pos
        self.left_foot_offset = np.array([0, 0.065, -0.175]) - com_pos

        self.right_leg_joint_ids = list(range(0, self.leg_dof))
        self.left_leg_joint_ids = list(range(self.leg_dof, self.leg_dof * 2))
        self.max_force_leg = [max_force] * self.leg_dof

        self.axis = np.array([
            [0, 0, 1], [1, 0, 0], [0, 1, 0],
            [0, 1, 0], [0, 1, 0], [1, 0, 0]
        ])
        self.identity = np.eye(3)

        self.left_leg_joints_value_logs = []
        self.right_leg_joints_value_logs = []

    def set_leg_joint_positions(self, joint_positions, leg='both'):
        if leg == 'right':
            p.setJointMotorControlArray(self.robot_id, self.right_leg_joint_ids,
                                        self.control_mode, targetPositions=joint_positions,
                                        forces=self.max_force_leg)
        elif leg == 'left':
            p.setJointMotorControlArray(self.robot_id, self.left_leg_joint_ids,
                                        self.control_mode, targetPositions=joint_positions,
                                        forces=self.max_force_leg)
        elif leg == 'both':
            self.set_leg_joint_positions(joint_positions, 'left')
            self.set_leg_joint_positions(joint_positions, 'right')
        else:
            raise ValueError("Invalid leg name. Choose 'left', 'right', or 'both'.")

    def set_leg_positions(self, left_pos, right_pos, rpy):
        ql = self.inverse_kinematics(left_pos, rpy, self.left_foot_offset)
        qr = self.inverse_kinematics(right_pos, rpy, self.right_foot_offset)

        # For Logging/graph purpose
        self.left_leg_joints_value_logs.append(ql)
        self.right_leg_joints_value_logs.append(qr)

        self.set_leg_joint_positions(ql, 'left')
        self.set_leg_joint_positions(qr, 'right')

    def enable_torque_mode(self):
        p.setJointMotorControlArray(self.robot_id,
                                    jointIndices=self.joint_ids,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0.0] * (self.leg_dof * 2))
        self.control_mode = p.TORQUE_CONTROL

    def get_leg_transforms(self, q, offset):
        zero = np.zeros(3)
        tf_ = tf.getTransFromRp

        T = []
        T.append(tf_(tf.rodriguesRotate(self.axis[0], q[0]), offset))
        for i in range(1, 6):
            trans = [0, 0, -self.l1] if i == 3 else ([0, 0, -self.l2] if i == 4 else zero)
            T.append(T[-1].dot(tf_(tf.rodriguesRotate(self.axis[i], q[i]), trans)))
        return T

    def forward_kinematics(self, q, offset):
        T = self.get_leg_transforms(q, offset)[-1]
        return tf.getRpFromT(T)

    def inverse_kinematics(self, p_ref, omega_ref, offset):
        q = self.get_joint_positions(offset)
        R, p = self.forward_kinematics(q, offset)
        omega = np.array(tf.getRPYFromR(R))
        error = np.append(p_ref - p, omega_ref - omega)
        dq = self._lambda * np.linalg.inv(self.jacobian(q, offset)) @ error
        return q + dq

    def jacobian(self, q, offset):
        T = self.get_leg_transforms(q, offset)
        p = [tf.getPFromT(t) for t in T]
        R = [tf.getRFromT(t) for t in T]
        wa = [R[i] @ self.axis[i] for i in range(6)]

        Jp = np.vstack([np.hstack((np.cross(wa[i], (p[5] - p[i])), wa[i])) for i in range(5)])
        J = np.vstack((Jp, np.hstack((np.zeros(3), wa[5])))).T
        return J

    def get_joint_positions(self, offset):
        if np.allclose(offset, self.right_foot_offset):
            joint_ids = self.right_leg_joint_ids
        elif np.allclose(offset, self.left_foot_offset):
            joint_ids = self.left_leg_joint_ids
        else:
            raise ValueError("Invalid leg offset.")

        return [p.getJointState(self.robot_id, j)[0] for j in joint_ids]

    def initialize_position(
        self,
        start_com_height=0.45,
        initial_rpy=[0, 0, 0],
        init_time=1.0,
        initial_joint_pos=[0.0, 0.0, -0.44, 0.88, -0.44, 0.0]
    ):
        """
        Smoothly initializes the robot in a standing pose by interpolating from zero to desired joint positions.
        This happens over `init_time` seconds.

        Args:
            start_com_height: Float, vertical distance of CoM from ground.
            initial_rpy: List of desired foot orientation (roll, pitch, yaw) in radians.
            init_time: Float, duration in seconds for the initialization process.
            initial_joint_pos: List of 6 floats, desired joint positions for both legs.

        """
        num_steps = int(init_time / self.env.time_step)
        step_values = np.linspace(0, 1, num_steps)

        # Define left/right foot target positions under the body
        foot_pos_r = [0.0, self.right_foot_offset[1], -start_com_height]
        foot_pos_l = [0.0, self.left_foot_offset[1], -start_com_height]

        # Phase 1: move to nominal joint positions directly
        for s in step_values:
            interp_pos = (1 - s) * np.zeros(6) + s * np.array(initial_joint_pos)
            self.set_leg_joint_positions(interp_pos.tolist(), 'left')
            self.set_leg_joint_positions(interp_pos.tolist(), 'right')
            self.reset_pose(
                pos=[0.0, 0.0, start_com_height + 0.02],
                orn=[0.0, 0.0, 0.0, 1.0]
            )
            self.env.step()

        # Phase 2: solve IK to adjust foot position while holding foot orientation
        for _ in range(num_steps):
            joint_pos_r = self.inverse_kinematics(foot_pos_r, initial_rpy, self.right_foot_offset)
            joint_pos_l = self.inverse_kinematics(foot_pos_l, initial_rpy, self.left_foot_offset)
            self.set_leg_joint_positions(joint_pos_l, 'left')
            self.set_leg_joint_positions(joint_pos_r, 'right')
            self.env.step()

    def disconnect(self):
        """
        Safely disconnects the simulation.
        """
        if p.isConnected():
            p.disconnect()
