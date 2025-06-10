import pybullet as p
import pybullet_data
import time

class Environment:
    def __init__(self, plane_path='plane.urdf', use_gui=True):
        # Connect to PyBullet
        p.connect(p.SHARED_MEMORY)
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Setup incline slider
        self.incline_slider = p.addUserDebugParameter("Incline", -0.1, 0.1, 0)
        self.plane_id = p.loadURDF(plane_path, [0, 0, 0], self._get_plane_quaternion())
        p.changeDynamics(self.plane_id, -1, lateralFriction=60)

        self.time_step = 1. / 240.
        p.setTimeStep(self.time_step)

    def _get_plane_quaternion(self):
        incline = p.readUserDebugParameter(self.incline_slider)
        return p.getQuaternionFromEuler([0, incline, 0])

    def reset_incline(self):
        p.resetBasePositionAndOrientation(self.plane_id, [0, 0, 0], self._get_plane_quaternion())

    def get_incline(self):
        return p.readUserDebugParameter(self.incline_slider)

    def step(self):
        p.stepSimulation()
        time.sleep(self.time_step)

    def set_camera_target(self, pos):
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=135, cameraPitch=-10, cameraTargetPosition=pos)

    def disconnect(self):
        p.disconnect()
