import numpy as np
import scipy.linalg as la
import control
from motion_planning.trajectory import TrajectoryGenerator

class PreviewControl:
    """
    PreviewControl implements a ZMP-based walking pattern generator
    using the Linear Inverted Pendulum Model (LIPM) and optimal preview control.
    It handles CoM trajectory generation, footstep planning, and foot trajectory creation.
    """
    def __init__(self, dt=1./240., Tsup_time=0.5, Tdl_time=0.1, CoM_height=0.45, g=9.81, previewStepNum=240,
                 initialTargetZMP=np.array([0.,0.]), R=np.matrix([1.]), Q=np.matrix([[7000,0,0,0],
                                                                                     [0,1,0,0],
                                                                                     [0,0,1,0],
                                                                                     [0,0,0,1]])):
        # Leg identifiers (for use in gait generation)
        self.RIGHT_LEG = 1
        self.LEFT_LEG = 0
        # Control loop time step 
        self.dt = dt
        # Number of future steps to preview (for ZMP planning)
        self.previewStepNum = previewStepNum

        # Define the Linear Inverted Pendulum Model (LIPM) dynamics in discrete-time:
        # A: state transition matrix (CoM pos, vel, acc)
        # B: control input matrix (jerk)
        # C: output matrix → maps CoM to ZMP using
        self.A = np.matrix([[1, dt, (dt**2) / 2],
                            [0, 1, dt],
                            [0, 0, 1]])
        self.B = np.matrix([(dt**3) / 6, (dt**2) / 2, dt]).T
        self.C = np.matrix([1, 0, -CoM_height / g])
        self.CoM_height = CoM_height

        # Initial state of the system for both x and y axes: 
        # [position, velocity, acceleration]
        self.x = np.matrix(np.zeros(3)).T
        self.y = np.matrix(np.zeros(3)).T

        # Preset array of alternating foot placements
        self.footPrints = np.array([[[0., 0.065], [0., -0.065]],
                                    [[0., 0.065], [0., -0.065]],
                                    [[0., 0.065], [0., -0.065]]])
        # Current ZMP estimates
        self.Tsup = int(Tsup_time / dt)
        # Number of steps per double support phase
        self.Tdl = int(Tdl_time / dt)

        # Desired ZMP trajectories (future reference)
        self.px_ref = np.full((self.Tsup + self.Tdl) * 3, initialTargetZMP[0])
        self.py_ref = np.full((self.Tsup + self.Tdl) * 3, initialTargetZMP[1])
        
        # Current ZMP estimates
        self.px = np.array([0.0])
        self.py = np.array([0.0])

        # Augmented state-space matrix
        # Includes ZMP tracking error and CoM dynamics
        # To optimize both ZMP tracking and smooth control (jerk minimization) over a preview horizon
        # ϕ = [[1, -CA], [0, A]]
        self.phi = np.hstack((np.matrix([1,0,0,0]).T, np.vstack((-self.C * self.A, self.A))))

        # Input matrix of the augmented system
        # Encodes how the control input (jerk) affects: 
        # ZMP tracking error (-CB) and CoM state (B)
        # G = [[-CB], [B]]
        self.G = np.vstack((-self.C * self.B, self.B))
        
        # Solve Discrete Algebraic Riccati Equation (DARE)
        # Solves the optimal control problem for minimizing CoM/ZMP error and control effort
        P, _, _ = control.dare(self.phi, self.G, Q, R)

        # Compute Preview Gain
        # Computes the closed-loop state transition matrix
        # (how future ZMP references propagate over the preview horizon)
        # ζ = (I − G*(R + G^T*P*G) − G^T*P)*ϕ
        zai = (np.eye(4) - self.G * la.inv(R + self.G.T*P*self.G) * self.G.T * P) * self.phi

        # Augmented Reference Input Matrix
        # Used to isolate the ZMP tracking error in the augmented state space
        # Where the full state is: x_aug = [[e_k], [x_k]]
        # Gr picks out the ZMP error e_k
        self.Gr = np.matrix([1., 0., 0., 0.,]).T

        # Preview Gain Vector
        # Array of preview gains for future ZMP references
        # Computed for each step in the preview horizon
        # (Computes the preview gains F_r(i) for all future steps i = 1,2,…,N)
        # Quantifies how much each future ZMP reference contributes to the current control input u_k
        # Fr(i) = -(R + G^T*P*G)*G.T*(ζ^T)^(i-1)*P*Gr
        self.Fr = np.array([])
        for i in range(1, previewStepNum + 1):
            self.Fr = np.append(self.Fr, -la.inv(R + self.G.T * P * self.G) * self.G.T * ((zai.T)**(i-1)) * P * self.Gr)

        # Feedback gain for the augmented state, derived from LQR solution
        # F = −(R + G^T*P*G) − G^T*P*ϕ
        # Governs how the controller reacts to the current ZMP tracking error and CoM state
        self.F = -la.inv(R + self.G.T * P * self.G) * self.G.T * P * self.phi

        # Reference ZMP Logs
        self.px_ref_log = self.px_ref[:(self.Tsup + self.Tdl) * 2]
        self.py_ref_log = self.py_ref[:(self.Tsup + self.Tdl) * 2]

        # Control and Update Variables
        # Used to store cumulative jerk and system updates over time
        # xdu is jerk applied in last iteration
        self.xdu = 0
        self.ydu = 0
        # cumulative jerk integrals
        self.xu = 0
        self.yu = 0

        # Derivatives of state for motion updates
        self.dx = np.matrix(np.zeros(3)).T
        self.dy = np.matrix(np.zeros(3)).T

        # Leg State and ZMP Tracking
        # Keeps track of swing/support leg
        self.swingLeg = self.RIGHT_LEG
        self.supportLeg = self.LEFT_LEG

        # For tracking past ZMP
        self.targetZMP_old = np.array([initialTargetZMP])
        
        self.currentFootStep = 0

        # Foot Trajectory Height
        # Maximum vertical height the foot reaches during a swing phase
        self.z_height = 0.4

    def footPrintAndCoM_trajectoryGenerator(self, inputTargetZMP, inputFootPrint, stepHeight=0.04):
        """
        Generates the Center of Mass (CoM) and foot trajectories for the next walking step 
        using ZMP preview control and kinematic foot planning.

        Parameters
        ----------
        inputTargetZMP : np.ndarray
            The target ZMP position [x, y] for the upcoming step.
        inputFootPrint : np.ndarray
            The desired footstep position [x, y] for the swing leg.
        stepHeight : float, optional
            Maximum vertical height of the foot during swing, by default 0.04 meters.

        Returns
        -------
        CoM_trajectory : np.ndarray
            The planned CoM trajectory over the preview window (T x 3), with fixed CoM height.
        leftTrj : np.ndarray
            The swing/support trajectory for the left foot (T x 3).
        rightTrj : np.ndarray
            The swing/support trajectory for the right foot (T x 3).

        Notes
        -----
        - Updates footstep queue and reference ZMP trajectory using FIFO.
        - Computes control input using preview control based on LIPM dynamics.
        - Uses the preview gain (Fr) and feedback gain (F) to track ZMP and generate smooth CoM motion.
        - Generates swing foot trajectory using polynomial or constant interpolation.
        - Alternates support/swing legs and logs old ZMP target.
        """

        currentFootStep = 0

        # Sets up the next footstep using the support leg and new inputFootPrint
        # Alternates between left and right using `self.supportLeg`
        # Updates self.footPrints which stores the alternating foot positions
        self.footPrints = self.footOneStep(self.footPrints, inputFootPrint, self.supportLeg)

        # Uses the current and previous ZMP targets to generate smooth reference ZMP trajectories 
        # for x and y axes during: double support phase (Tdl) & single support phase (Tsup)
        input_px_ref, input_py_ref = self.targetZMP_generator(inputTargetZMP, \
                                                              self.targetZMP_old[-1], \
                                                              self.Tsup, self.Tdl)

        # Updates the existing ZMP reference buffer with new ZMP references
        self.px_ref = self.fifo(self.px_ref, input_px_ref, len(input_px_ref))
        self.py_ref = self.fifo(self.py_ref, input_py_ref, len(input_py_ref))

        # Logs the ZMP reference history
        self.px_ref_log = np.append(self.px_ref_log, input_px_ref)
        self.py_ref_log = np.append(self.py_ref_log, input_py_ref)

        # Preview Control Loop
        # Computes the CoM trajectory point-by-point using the preview control formulation
        #########################################
        CoM_trajectory = np.empty((0, 3), float)

        for i in range(len(input_px_ref)):
            # Step 1: Compute ZMP difference
            dpx_ref = self.px_ref[i + 1] - self.px_ref[i]
            dpy_ref = self.py_ref[i + 1] - self.py_ref[i]

            # Step 2: Compute current ZMP error
            # self.C * self.x gives predicted ZMP from current CoM state
            # e(t) = p_ref(t) − C*x(t)
            xe = self.px_ref[i] - self.C * self.x
            ye = self.py_ref[i] - self.C * self.y

            # Step 3: Extended state for LQR (including previous jerk effect)
            X = self.phi * np.vstack((xe, self.dx)) + self.G * self.xdu + self.Gr * dpx_ref
            Y = self.phi * np.vstack((ye, self.dy)) + self.G * self.ydu + self.Gr * dpy_ref

            # Step 4: Preview Sum
            # Accumulates the feedforward correction for the future desired ZMP changes
            # px_ref[i+j]−px_ref[i+j−1] is the change in the desired ZMP at step i+j
            # the optimal control input includes both:
            # 1. Feedback on the current state (F⋅X)
            # 2. Feedforward correction over a preview horizon (∑Fr(j)⋅Δp_ref)
            # This optimally balances tracking error and control effort over the preview horizon.
            # Total jerk command: u(i) = −F*X(i) + ∑Fr(j)⋅(p_ref(i+j)−p_ref(i+j−1))|(j=1...N)
            # u(i) is the jerk input at time i, F is the feedback gain, Fr(j) is the preview gain for step j, pref is the reference ZMP
            # so, xsum and ysum are the cumulative feedforward terms for x and y directions respectively
            xsum = ysum = 0
            for j in range(1, self.previewStepNum + 1):
                xsum += self.Fr[j - 1] * (self.px_ref[i + j] - self.px_ref[i + j - 1])
                ysum += self.Fr[j - 1] * (self.py_ref[i + j] - self.py_ref[i + j - 1])

            # Step 5: Compute Final Jerk Command
            self.xdu = self.F * X + xsum
            self.ydu = self.F * Y + ysum

            # Step 6: Integrate Jerk to Get Acceleration
            # Integrating jerk gives acceleration input to the discrete Linear Inverted Pendulum Model (LIPM)
            self.xu += self.xdu
            self.yu += self.ydu

            # Step 7: Update CoM State
            # Updates the state vector (position, velocity, acceleration) using discrete-time dynamics
            # x(k+1) = Ax_k + Bu_k (x∈R^3: CoM pos, vel, accn, u_k: accn (integrated jerk))
            old_x = self.x
            old_y = self.y

            self.x = self.A * self.x + self.B * self.xu
            self.y = self.A * self.y + self.B * self.yu

            # Step 8: Estimate Derivative of State
            # Estimates how much the state changed since the last timestep
            # Used in the next iteration to construct the extended state vector [[e], [xdot]]
            self.dx = self.x - old_x
            self.dy = self.y - old_y

            # Step 9: Log CoM Trajectory
            # Logs the current center of mass (CoM) position into a trajectory list
            # The height remains constant z=CoM_heightz, assuming LIPM constraints
            CoM_trajectory = np.vstack((CoM_trajectory, [self.x[0, 0], self.y[0, 0], self.CoM_height]))

            # Step 10: Compute ZMP from CoM
            # Estimates the Zero Moment Point (ZMP) from CoM using
            # ZMP = C*x = x - (z/g)*xddot
            self.px = np.append(self.px, self.C * self.x)
            self.py = np.append(self.py, self.C * self.y)

        # Step 11: Generate Foot Trajectory
        # Generates the 3D foot swing and support trajectories for the current gait step
        leftTrj, rightTrj = self.footTrajectoryGenerator(np.hstack((self.footPrints[currentFootStep, self.swingLeg], 0.)),
                                                         np.hstack((self.footPrints[currentFootStep + 1, self.swingLeg], 0.)),
                                                         np.array([0., 0., 0.]),
                                                         np.array([0., 0., 0.]),
                                                         np.hstack((self.footPrints[currentFootStep, self.supportLeg], 0.)),
                                                         self.swingLeg,
                                                         stepHeight=stepHeight)

        # Step 12: Switch Swing and Support Legs
        # Updates the state: the leg that just swung becomes the new support leg, and vice versa
        # Helps alternate gait with each CoM phase
        self.swingLeg, self.supportLeg = self.changeSupportLeg(self.swingLeg, self.supportLeg)
        
        # Step 13: Update ZMP History
        # Stores the current target ZMP as the latest "old" one
        # To ensure smooth transitions in the next step generation via interpolation
        self.targetZMP_old = np.vstack((self.targetZMP_old, inputTargetZMP))

        return CoM_trajectory, leftTrj, rightTrj

    def targetZMP_generator(self, targetZMP, targetZMP_old, Tsup, Tdl):
        tdl_t = np.arange(0, Tdl)

        x_a = (targetZMP_old[0] - targetZMP[0]) / (0 - Tdl)
        x_b = targetZMP_old[0]
        y_a = (targetZMP_old[1] - targetZMP[1]) / (0 - Tdl)
        y_b = targetZMP_old[1]

        px_ref = np.hstack((x_a * tdl_t + x_b, np.full(Tsup, targetZMP[0])))
        py_ref = np.hstack((y_a * tdl_t + y_b, np.full(Tsup, targetZMP[1])))
        return px_ref, py_ref

    def footTrajectoryGenerator(self, startPointV, endPointV, startRobotVelocityV, endRobotVelocityV,
                                supportPointV, swingLeg, stepHeight=0.4):
        """
        Generates swing and support foot trajectories for one walking step using polynomial interpolation.

        Parameters
        ----------
        startPointV : np.ndarray
            3D start position [x, y, z] of the swing foot.
        endPointV : np.ndarray
            3D end (landing) position [x, y, z] of the swing foot.
        startRobotVelocityV : np.ndarray
            Initial velocity of the robot's swing foot [vx, vy, vz].
        endRobotVelocityV : np.ndarray
            Final velocity of the robot's swing foot [vx, vy, vz].
        supportPointV : np.ndarray
            3D fixed position [x, y, z] of the support foot.
        swingLeg : int
            Identifier for the swing leg (self.LEFT_LEG or self.RIGHT_LEG).
        stepHeight : float, optional
            Maximum height of the foot during swing phase (default is 0.4 m).

        Returns
        -------
        trjL : np.ndarray
            Left foot trajectory (T x 3), either as swing or support.
        trjR : np.ndarray
            Right foot trajectory (T x 3), either as swing or support.

        Notes
        -----
        - The foot trajectory is split into two phases: double support (Tdl) and single support (Tsup).
        - During double support, the swing foot is stationary at the start point.
        - If the foot is not moving (i.e., start == end), a constant trajectory is used.
        - Otherwise, a smooth trajectory is generated using a polynomial interpolator.
        - The support foot remains fixed throughout the step.
        """
        # Generate trajectory for the support foot (stationary throughout the step)
        supportTrj = np.vstack((np.full(self.Tdl + self.Tsup, supportPointV[0]),     # X
                                np.full(self.Tdl + self.Tsup, supportPointV[1]),     # Y
                                np.full(self.Tdl + self.Tsup, supportPointV[2]))).T  # Z

        # Generate constant trajectory during double support for the swing foot
        trajectoryForTdl = np.vstack((np.full(self.Tdl, startPointV[0]),
                                      np.full(self.Tdl, startPointV[1]),
                                      np.full(self.Tdl, startPointV[2]))).T

        # If the start and end points are the same, use constant position for the full Tsup duration
        if np.array_equal(startPointV, endPointV):
            trajectoryForTsup = np.vstack((np.full(self.Tsup, startPointV[0]),
                                           np.full(self.Tsup, startPointV[1]),
                                           np.full(self.Tsup, startPointV[2]))).T
        else:
            # Generate swing trajectory using polynomial interpolation
            trajectoryForTsup = TrajectoryGenerator(startPointV, endPointV, 
                                                    -startRobotVelocityV, -endRobotVelocityV,
                                                    stepHeight, 
                                                    0., self.Tsup * self.dt, 
                                                    self.dt,
                                                    method="poly").generate()
        # Combine the two phases to form full trajectory for swing and support legs
        trjL = None
        trjR = None
        if swingLeg is self.RIGHT_LEG:
            # Right leg is swinging; left leg is support
            trjR = np.vstack((trajectoryForTdl, trajectoryForTsup))
            trjL = supportTrj
        elif swingLeg is self.LEFT_LEG:
            # # Left leg is swinging; right leg is support
            trjL = np.vstack((trajectoryForTdl, trajectoryForTsup))
            trjR = supportTrj

        return trjL, trjR

    def fifo(self, p, in_p, range, vstack=False):
        """
        Implements a FIFO (First-In-First-Out) buffer update.

        Parameters
        ----------
        p : np.ndarray
            Existing buffer (1D or 2D array).
        in_p : np.ndarray
            New data to append to the buffer.
        range : int
            Number of old elements to remove from the front.
        vstack : bool, optional
            Whether to vertically stack the arrays (2D arrays), by default False.

        Returns
        -------
        np.ndarray
            Updated buffer with old elements removed and new ones appended.

        Notes
        -----
        - If `vstack` is True, assumes input is a 2D matrix and uses `np.vstack`.
        - Otherwise, uses `np.append` for 1D buffer updates.
        """
        if vstack:
            # Delete first `range` rows and append new rows
            return np.vstack((np.delete(p, range, 0), in_p))
        else:
            # Delete first `range` elements and append new elements
            return np.append(np.delete(p, slice(range), None), in_p)

    def footOneStep(self, footPrints, supportPoint, supportLeg):
        """
        Appends the next foot placement to the existing footprint plan.

        Parameters
        ----------
        footPrints : np.ndarray
            Array of foot positions for previous steps. Shape: (N, 2, 2)
        supportPoint : np.ndarray
            The new foot location (x, y) for the next step.
        supportLeg : int
            The leg currently acting as support (LEFT_LEG or RIGHT_LEG).

        Returns
        -------
        np.ndarray
            Updated array of foot placements with the new step added.

        Notes
        -----
        - Keeps the non-stepping foot fixed and updates only the swing foot.
        - Removes the first element (oldest) to maintain constant horizon length.
        """
        newFootPrint = None

        if supportLeg is self.LEFT_LEG:
            # # Left leg is support -> swing right leg
            newFootPrint = np.vstack((footPrints, 
                                    [np.vstack((supportPoint, footPrints[-1, 1]))] # right leg is reused
                                    ))
        elif supportLeg is self.RIGHT_LEG:
            # # Right leg is support -> swing left leg
            newFootPrint = np.vstack((footPrints, 
                                    [np.vstack((footPrints[-1, 0], supportPoint))] # left leg is reused
                                    ))
        # Remove the oldest footstep to maintain buffer size
        return np.delete(newFootPrint, 0, 0)

    def changeSupportLeg(self, swingLeg, supportLeg):
        """
        Swaps the roles of the swing and support leg.

        Parameters
        ----------
        swingLeg : int
            Current swing leg identifier.
        supportLeg : int
            Current support leg identifier.

        Returns
        -------
        Tuple[int, int]
            New (swingLeg, supportLeg) pair with roles swapped.

        Notes
        -----
        - Useful for alternating foot placement during walking.
        """
        return supportLeg, swingLeg