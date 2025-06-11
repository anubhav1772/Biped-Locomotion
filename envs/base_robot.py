import pybullet as p
from envs.environment import Environment

class Robot:
    def __init__(self, urdf_path, start_pos, start_orn, max_force, control_mode=p.POSITION_CONTROL):
        self.env = Environment()
        self.robot_id = p.loadURDF(urdf_path, start_pos, p.getQuaternionFromEuler(start_orn))

        self.control_mode = control_mode
        self.max_force = max_force
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_ids = list(range(self.num_joints))
        self.max_force_list = [max_force] * self.num_joints

        for link in range(-1, self.num_joints):
            p.changeVisualShape(self.robot_id, link, rgbaColor=[0.5, 0.5, 0, 1])

        # Robot-specific sliders
        self.stride_slider = p.addUserDebugParameter('Stride', 0, 0.2, 0.1)
        self.step_height_slider = p.addUserDebugParameter('Step height', 0.03, 0.1, 0.04)

    def get_euler(self):
        _, qua = p.getBasePositionAndOrientation(self.robot_id)
        return p.getEulerFromQuaternion(qua)

    def get_quaternion(self):
        _, qua = p.getBasePositionAndOrientation(self.robot_id)
        return qua

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return pos

    def get_stride(self):
        return p.readUserDebugParameter(self.stride_slider)

    def get_step_height(self):
        return p.readUserDebugParameter(self.step_height_slider)

    def reset_pose(self, pos, orn):
        p.resetBasePositionAndOrientation(self.robot_id, pos, orn)

    def set_motor_torque(self, target_torques):
        if self.control_mode != p.TORQUE_CONTROL:
            print("Error: Control mode must be TORQUE_CONTROL.")
            return
        p.setJointMotorControlArray(
            self.robot_id, jointIndices=self.joint_ids,
            controlMode=p.TORQUE_CONTROL, forces=target_torques
        )

    def set_motor_position(self, target_positions):
        p.setJointMotorControlArray(
            self.robot_id, jointIndices=self.joint_ids,
            controlMode=self.control_mode, targetPositions=target_positions,
            forces=self.max_force_list
        )

