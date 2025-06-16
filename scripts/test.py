from envs.environment import Environment
from envs.base_robot import Robot
import pybullet as p

def main():
    # print(p.getConnectionInfo())
    # Initialize environment and robot
    # env = Environment() # already being called in Robot
    # control_mode=p.POSITION_CONTROL (by default)
    robot = Robot(
        urdf_path="./urdf/biped_flatfoot_robot.urdf",
        start_pos=[0, 0, 0.4],
        start_orn=[0, 0, 0],
        max_force=9.0
    ) 

    try:
        for _ in range(10000):
            target_positions = [0.0] * robot.num_joints
            robot.set_motor_position(target_positions)

            # Camera follows the robot
            robot.env.set_camera_target(robot.get_position())

            # Step simulation
            robot.env.step()

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    finally:
        robot.env.disconnect()

if __name__ == "__main__":
    main()
