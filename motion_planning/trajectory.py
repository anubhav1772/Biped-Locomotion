import numpy as np
from scipy.interpolate import CubicSpline
import scipy.linalg as la

class TrajectoryGenerator:
    def __init__(self, 
                start_point, 
                end_point, 
                start_vel_xy, 
                end_vel, 
                z_height, 
                start_time, 
                end_time, 
                dt,
                method="spline"):
        """
        Initialize the 3D trajectory generator.

        Args:
            start_point (np.array): Initial position in 3D space (shape: (3,))
            end_point (np.array): Final position in 3D space (shape: (3,))
            start_vel_xy (np.array): Initial velocities in x and y (shape: (2,))
            end_vel (np.array): Final velocities in x, y, and z (shape: (3,))
            z_height (float): Height of the peak in the z direction (used for creating an arc)
            start_time (float): Start time of the trajectory
            end_time (float): End time of the trajectory
            dt (float): Time step for sampling
            method (str): Type of trajectory generation method - "spline" (default) or "poly"
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.start_vel_xy = np.array(start_vel_xy)
        self.end_vel = np.array(end_vel)
        self.z_height = z_height
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt

        self.method = method
        
        # Time vector
        self.time_vec = np.arange(self.start_time, self.end_time, self.dt)
    
    def generate(self):
        """
        Generate the full 3D trajectory as a Nx3 numpy array.
        
        Returns:
            np.ndarray: Trajectory array of shape (N, 3) where N = number of time steps
        """
        if self.method == "spline": 
            # Generate x and y using clamped cubic spline with boundary velocities
            x = self._cubic_spline_trajectory_generator_xy(self.start_point[0], self.end_point[0],
                                                           self.start_vel_xy[0], self.end_vel[0])
            y = self._cubic_spline_trajectory_generator_xy(self.start_point[1], self.end_point[1],
                                                           self.start_vel_xy[1], self.end_vel[1])
        else:
            # Generate x and y using cubic polynomial interpolation
            x = self._cubic_poly_trajectory_generator_xy(self.start_point[0], self.end_point[0],
                                                         self.start_vel_xy[0], self.end_vel[0],
                                                         self.start_time, self.end_time)
            y = self._cubic_poly_trajectory_generator_xy(self.start_point[1], self.end_point[1],
                                                     self.start_vel_xy[1], self.end_vel[1],
                                                     self.start_time, self.end_time)
        # z is always generated using a peak trajectory
        z = self._trajectory_generator_z(self.z_height, 
                                         self.start_point[2], self.end_point[2],
                                         self.end_vel[2])
        # Stack x, y, z trajectories column-wise to form (N, 3)
        return np.vstack((x, y, z)).T
    
    def _cubic_spline_trajectory_generator_xy(self, 
                                              start_pos, end_pos, 
                                              start_vel, end_vel):
        """
        Generate cubic spline trajectory for x or y using boundary conditions (clamped spline).

        Args:
            start_pos (float): Start position
            end_pos (float): End position
            start_vel (float): Start velocity
            end_vel (float): End velocity

        Returns:
            np.ndarray: Interpolated trajectory values over self.time_vec
        """
        times = np.array([self.start_time, self.end_time])
        positions = np.array([start_pos, end_pos])

        # Use CubicSpline with clamped boundary conditions (first derivatives)
        cs = CubicSpline(times, positions, bc_type=((1, start_vel), (1, end_vel)))
        
        return cs(self.time_vec)

    def _cubic_poly_trajectory_generator_xy(self, 
                                        start_pos, end_pos, 
                                        start_vel, end_vel, 
                                        start_time, end_time):
        """
        Generate cubic polynomial trajectory for x or y using interpolation and velocity constraints.

        Args:
            start_pos (float): Start position
            end_pos (float): End position
            start_vel (float): Start velocity
            end_vel (float): End velocity
            start_time (float): Start time
            end_time (float): End time

        Returns:
            np.ndarray: Interpolated trajectory values over self.time_vec
        """
         # Right-hand side vector A for the system B * C = A (shape: [4,1])
        A = np.array([[start_pos], [end_pos], [start_vel], [end_vel]])

        # Construct matrix B from the boundary conditions (position and velocity) 
        # 4x4 matrix
        B = np.array([
            [start_time**3, start_time**2, start_time, 1],
            [end_time**3,   end_time**2,   end_time,   1],
            [3*start_time**2, 2*start_time, 1, 0],
            [3*end_time**2,   2*end_time,   1, 0]
        ])

        # Solve for polynomial coefficients C
        # C = np.linalg.solve(B, A)
        C = la.inv(B) @ A
          
        # print(A.shape) #(4,1)
        # print(B.shape) #(4,4)
        # print(C.shape) #(4,4)

        # Evaluate the cubic polynomial using the coefficients
        x = C[0] * self.time_vec**3 + C[1] * self.time_vec**2 + C[2] * self.time_vec + C[3]
        return x  # 2D array (4,1)

    def _trajectory_generator_z(self, 
                                z_height, 
                                start_pos, end_pos, 
                                end_vel):
        """
        Generate a z-axis trajectory with a smooth peak using cubic polynomial interpolation.
        Ensures the trajectory starts at start_pos, ends at end_pos with a specified peak height,
        and meets end velocity condition.

        Args:
            z_height (float): Height of the peak relative to the start_pos
            start_pos (float): Starting z position
            end_pos (float): Ending z position
            end_vel (float): Final z velocity

        Returns:
            np.ndarray: z-values over time forming a curved path with a peak
        """
        # Midpoint in time where the trajectory reaches the peak
        height_time = (self.end_time + self.start_time) / 2
        peak_height = start_pos + z_height
        
        # Boundary conditions vector A
        A = np.array([peak_height, end_pos, start_pos, end_vel])

        # Coefficient matrix B for cubic interpolation using peak constraint and boundary conditions
        B = np.array([
            [height_time**3, height_time**2, height_time, 1],
            [self.end_time**3, self.end_time**2, self.end_time, 1],
            [self.start_time**3, self.start_time**2, self.start_time, 1],
            [3 * self.end_time**2, 2 * self.end_time, 1, 0]
        ])
        
        C = np.linalg.solve(B, A)

        # Evaluate cubic polynomial
        return C[0] * self.time_vec**3 + C[1] * self.time_vec**2 + C[2] * self.time_vec + C[3]

if __name__ == "__main__":
    # Test 1: Spline trajectory
    spline_traj_gen = TrajectoryGenerator(
        start_point=[0, 0, 0],
        end_point=[10, 5, 0],
        start_vel_xy=[0, 0],
        end_vel=[0, 0, 0],
        z_height=2,
        start_time=0,
        end_time=5,
        dt=0.01)

    trajectory = spline_traj_gen.generate()
    print("Method:", spline_traj_gen.method)
    print("Trajectory shape:", trajectory.shape)  # (N, 3)
    print("Trajectory (first 5 points):\n", trajectory[:5])

    # Test 2: Polynomial trajectory
    poly_traj_gen = TrajectoryGenerator(
        start_point=[0, 0, 0],
        end_point=[10, 5, 0],
        start_vel_xy=[0, 0],
        end_vel=[0, 0, 0],
        z_height=2,
        start_time=0,
        end_time=5,
        dt=0.01,
        method="poly")

    trajectory = poly_traj_gen.generate()
    print("\nMethod:", poly_traj_gen.method)
    print("Trajectory shape:", trajectory.shape)  # (N, 3)
    print("Trajectory (first 5 points):\n", trajectory[:5])
