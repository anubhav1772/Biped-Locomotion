import numpy as np
from typing import Tuple

class TransformUtils:
    @staticmethod
    def getTransFromRp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Construct a 4x4 transformation matrix from a rotation matrix and translation vector.
        """
        T = np.vstack((
            np.hstack((R, np.array(p).reshape(3, 1))),
            np.array([[0, 0, 0, 1]])
        ))
        return T

    @staticmethod
    def getRotationRoll(theta: float) -> np.ndarray:
        """Rotation about X-axis (roll)"""
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

    @staticmethod
    def getRotationPitch(theta: float) -> np.ndarray:
        """Rotation about Y-axis (pitch)"""
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    @staticmethod
    def getRotationYaw(theta: float) -> np.ndarray:
        """Rotation about Z-axis (yaw)"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def getPFromT(T: np.ndarray) -> np.ndarray:
        """Extract position vector from transformation matrix"""
        return T[0:3, 3]

    @staticmethod
    def getRFromT(T: np.ndarray) -> np.ndarray:
        """Extract rotation matrix from transformation matrix"""
        return T[0:3, 0:3]

    @staticmethod
    def getRpFromT(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract rotation matrix and position vector"""
        p = T[0:3, 3]
        R = T[0:3, 0:3]
        return R, p

    @staticmethod
    def getRPYFromR(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to roll, pitch, yaw (radians).
        Assumes the rotation order is ZYX.
        """
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw])

    @staticmethod
    def skewSymmetricMatrix(v: np.ndarray) -> np.ndarray:
        """Return the skew-symmetric matrix of a vector"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def rodriguesRotate(a: np.ndarray, theta: float) -> np.ndarray:
        """
        Rotate E about axis a (unit vector) by angle θ using Rodrigues' formula.

        Rodrigues' formula (Matrix Form):
            R = I + sinθ⋅K + (1 − cosθ)⋅K^2
        Where
            R is the rotation matrix, can be used to rotate any vector or matrix,
            K is the skew-symmetric matrix corresponding to the 3D axis vector 
                a = [ax,ay,az]^T
            K = [[0, -az, ay], 
                 [az, 0, -ax], 
                 [-ay, ax, 0]]
            K⋅v = a×v (i.e. multiplying K by a vector v is the same as computing the cross product a×v)
        """
        I = np.eye(3)
        K = TransformUtils.skewSymmetricMatrix(a)
        R = I + np.sin(theta) * K  + (1 - np.cos(theta)) * (K @ K)
        return R
