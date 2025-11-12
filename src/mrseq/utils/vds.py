"""Functions to generate a variable density spiral trajectory.

Program translated from the matlab program of Brian Hargreaves:
http://mrsrl.stanford.edu/~brian/vdspiral/

Description given by Brian Hargreaves

    function [k,g,s,time,r,theta] = vds(smax,gmax,T,N,Fcoeff,rmax)

    VARIABLE DENSITY SPIRAL GENERATION:
    ----------------------------------

    Function generates variable density spiral which traces
    out the trajectory

            k(t) = r(t) exp(i*q(t)), 		[1]

    Where q is the same as theta...
        r and q are chosen to satisfy:

        1) Maximum gradient amplitudes and slew rates.
        2) Maximum gradient due to FOV, where FOV can
           vary with k-space radius r/rmax, as

            FOV(r) = Sum    Fcoeff(k)*(r/rmax)^(k-1)   [2]


    INPUTS:
    -------
    smax = maximum slew rate in Hz/m/s
    gmax = maximum gradient in Hz/m (limited by Gmax or FOV)
    T = sampling period (s) for gradient AND acquisition.
    N = number of interleaves.
    Fcoeff = FOV coefficients with respect to r - see above.
    rmax= value of k-space radius at which to stop (m^-1).
        rmax = 1/(2*resolution)


    OUTPUTS:
    --------
    k = k-space trajectory (kx+iky) in m-1.
    g = gradient waveform (Gx+iGy) in Hz/m.
    s = derivative of g (Sx+iSy) in Hz/m/s.
    time = time points corresponding to above (s).
    r = k-space radius vs time (used to design spiral)
    theta = atan2(ky,kx) = k-space angle vs time.

Methods
-------
    Let r1 and r2 be the first derivatives of r in [1].
    Let q1 and q2 be the first derivatives of theta in [1].
    Also, r0 = r, and q0 = theta - sometimes both are used.
    F = F(r) defined by Fcoeff.

    Differentiating [1], we can get G = a(r0,r1,q0,q1,F)
    and differentiating again, we get S = b(r0,r1,r2,q0,q1,q2,F)

    (functions a() and b() are reasonably easy to obtain.)

    FOV limits put a constraint between r and q:

        dr/dq = N/(2*pi*F)				[3]

    We can use [3] and the chain rule to give

        q1 = 2*pi*F/N * r1				[4]

    and

        q2 = 2*pi/N*dF/dr*r1^2 + 2*pi*F/N*r2		[5]



    Now using [4] and [5], we can substitute for q1 and q2
    in functions a() and b(), giving

        G = c(r0,r1,F)
    and 	S = d(r0,r1,r2,F,dF/dr)

    Using the fact that the spiral should be either limited
    by amplitude (Gradient or FOV limit) or slew rate, we can
    solve
        |c(r0,r1,F)| = |Gmax|  				[6]

    analytically for r1, or

        |d(r0,r1,r2,F,dF/dr)| = |Smax|	 		[7]

    analytically for r2.

    [7] is a quadratic equation in r2.  The smaller of the
    roots is taken, and the np.real part of the root is used to
    avoid possible numeric errors - the roots should be np.real
    always.

    The choice of whether or not to use [6] or [7], and the
    solving for r2 or r1 is done by findq2r2 - in this .m file.

    Once the second derivative of theta(q) or r is obtained,
    it can be integrated to give q1 and r1, and then integrated
    again to give q and r.  The gradient waveforms follow from
    q and r.

    Brian Hargreaves -- Sept 2000.

    See Brian's journal, Vol 6, P.24.
"""

import numpy as np
import pypulseq as pp


def quadratic_formula_solver(a: float, b: float, c: float) -> tuple[float, float]:
    """Return the roots of a quadratic equation in the form ax^2 + bx + c = 0.

    Parameters
    ----------
    a
        Coefficient of the x^2 term.
    b
        Coefficient of the x term.
    c
        Constant term of the equation.

    Returns
    -------
    tuple(float, float)
        The two roots (solutions) of the quadratic equation.
    """
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:  # we only take the real part
        root1 = -b / (2 * a)
        root2 = -b / (2 * a)
    else:
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)

    return root1, root2


def calculate_angular_and_radial_acceleration(
    max_slew: float,
    max_grad: float,
    radius: float,
    radius_derivative: float,
    sampling_period: float,
    sampling_period_os: float,
    n_interleaves: int,
    fov_coefficients: list,
    max_kspace_radius: float,
) -> tuple[float, float]:
    """Calculate second derivatives of angle (theta) and radius (r) for a VDS trajectory.

    Parameters
    ----------
    max_slew
        Maximum slew rate of the system in Hz/m/s.
    max_grad
        Maximum gradient amplitude in Hz/m.
    radius
        Current radius of the spiral being constructed in meters.
    radius_derivative
        Derivative of the radius (rate of change) of the spiral in meters.
    sampling_period
        Sampling period (s) for gradient and acquisition.
    sampling_period_os
        Sampling period (s) for gradient and acquisition, divided by an oversampling factor.
    n_interleaves
        Number of spiral arms (interleaves).
    fov_coefficients
        List of coefficients defining the Field of View (FOV) profile.
    max_kspace_radius
        Maximum radius in k-space in m^(-1).

    Returns
    -------
    tuple[float, float]
        Angular acceleration (q2) in rad/s^(-2) and radial acceleration (r2) in m/s^(-2).
    """
    # Initialize Field of View (fov) value and its derivative
    fov = 0
    fov_derivative = 0

    # Calculate fov and its derivative based on radius and fov_coefficients
    for index, coefficient in enumerate(fov_coefficients):
        fov += coefficient * (radius / max_kspace_radius) ** index
        if index > 0:
            fov_derivative += index * coefficient * (radius / max_kspace_radius) ** (index - 1) / max_kspace_radius

    # Determine adjusted maximum gradient amplitude based on fov
    max_grad_for_fov = 1 / fov / sampling_period_os
    adjusted_max_gradient = min(max_grad_for_fov, max_grad)

    # Limit radius derivative based on adjusted maximum gradient
    max_radius_derivative = np.sqrt(adjusted_max_gradient**2 / (1 + (2 * np.pi * fov * radius / n_interleaves) ** 2))

    # Determine radial acceleration based on gradient limit
    if radius_derivative > max_radius_derivative:
        # Adjust radial acceleration to stay within max gradient amplitude
        radial_acceleration = (max_radius_derivative - radius_derivative) / sampling_period
    else:
        # Calculate frequency values for angular acceleration calculation
        angular_freq_over_interleaves = 2 * np.pi * fov / n_interleaves
        angular_freq_squared = angular_freq_over_interleaves**2

        # Calculate coefficients for radial acceleration equation under maximum slew rate constraint
        a = radius**2 * angular_freq_squared + 1
        b = (
            2 * angular_freq_squared * radius * radius_derivative**2
            + 2 * angular_freq_squared / fov * fov_derivative * radius**2 * radius_derivative**2
        )
        c = (
            angular_freq_squared**2 * radius**2 * radius_derivative**4
            + 4 * angular_freq_squared * radius_derivative**4
            + (2 * np.pi / n_interleaves * fov_derivative) ** 2 * radius**2 * radius_derivative**4
            + 4 * angular_freq_squared / fov * fov_derivative * radius * radius_derivative**4
            - max_slew**2
        )

        # Solve for radial acceleration (r2)
        roots = quadratic_formula_solver(a, b, c)
        radial_acceleration = np.real(roots[0])

        # Calculate actual slew rate and check for violations
        _tmp1 = 1j * angular_freq_over_interleaves
        _tmp2 = (
            2 * radius_derivative**2
            + radius * radial_acceleration
            + fov_derivative / fov * radius * radius_derivative**2
        )

        slew_rate_vector = radial_acceleration - angular_freq_squared * radius * radius_derivative**2 + _tmp1 * _tmp2
        slew_rate_ratio = np.abs(slew_rate_vector) / max_slew

        # Print warning if slew rate violation detected
        if slew_rate_ratio > 1.0 + 1e-6:
            raise ValueError(
                f'Slew rate violation detected for radius = {radius}.\n'
                f'Current slew rate = {round(np.abs(slew_rate_vector))} '
                f'Maximum slew rate = {round(max_slew)} (ratio = {round(slew_rate_ratio, 2)})\n'
            )

    # Calculate angular acceleration (q2)
    angular_acceleration = (
        2 * np.pi / n_interleaves * fov_derivative * radius_derivative**2
        + 2 * np.pi * fov / n_interleaves * radial_acceleration
    )

    return angular_acceleration, radial_acceleration


def variable_density_spiral_trajectory(
    system: pp.Opts,
    sampling_period: float,
    n_interleaves: int,
    fov_coefficients: list,
    max_kspace_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a variable density spiral (VDS) trajectory.

    Parameters
    ----------
    system
        PyPulseq system object containing gradient and slew rate limits.
    sampling_period
        Base sampling period for gradient and acquisition.
    n_interleaves
        Number of spiral arms (interleaves) in the trajectory.
    fov_coefficients
        Coefficients defining the Field of View (FOV) profile.
    max_kspace_radius
        Maximum k-space radius in inverse meters.

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        - k-space trajectory (k)
        - Gradient waveform (g)
        - Slew rate (s)
        - Time points for the trajectory (time)
        - Radius values (r)
        - Angular positions (theta)
    """
    # Extract system limits from PyPulseq system object
    max_slew = system.max_slew * 0.9
    max_grad = system.max_grad * 0.9

    # Define oversampling factor for finer time resolution during calculations
    oversampling_factor = 12
    sampling_period_os = sampling_period / oversampling_factor

    # Initialize angular and radial positions and derivatives
    angular_position = 0
    angular_velocity = 0
    radial_position = 0
    radial_velocity = 0

    # Initialize lists for storing the trajectory
    angular_positions = [angular_position]
    radial_positions = [radial_position]

    while radial_position < max_kspace_radius:
        q2, r2 = calculate_angular_and_radial_acceleration(
            max_slew,
            max_grad,
            radial_position,
            radial_velocity,
            sampling_period_os,
            sampling_period,
            n_interleaves,
            fov_coefficients,
            max_kspace_radius,
        )

        # Integrate for r, r', theta and theta'
        angular_velocity += q2 * sampling_period_os
        angular_position += angular_velocity * sampling_period_os

        radial_velocity += r2 * sampling_period_os
        radial_position += radial_velocity * sampling_period_os

        # Append current positions to the lists
        angular_positions.append(angular_position)
        radial_positions.append(radial_position)

    # Determine the number of points in the trajectory
    n_points = len(radial_positions)

    # Convert lists to numpy arrays with shape (n_points, 1)
    angular_positions = np.array(angular_positions)[:, np.newaxis]
    radial_positions = np.array(radial_positions)[:, np.newaxis]
    time_points = np.arange(n_points)[:, np.newaxis] * sampling_period_os

    # Downsample trajectory to original sampling period
    downsample_indices = slice(round(oversampling_factor / 2), n_points, oversampling_factor)
    radial_positions = radial_positions[downsample_indices]
    angular_positions = angular_positions[downsample_indices]
    time_points = time_points[downsample_indices]

    # Adjust length of arrays to be a multiple of 4
    valid_length = 4 * (len(angular_positions) // 4)
    radial_positions = radial_positions[:valid_length]
    angular_positions = angular_positions[:valid_length]
    time_points = time_points[:valid_length]

    # Compute k-space trajectory on the regular time raster
    k_space_trajectory = radial_positions * np.exp(1j * angular_positions)

    # Calculate gradient waveform by shifting k-space trajectory
    k_shifted_forward = np.vstack([np.zeros((1, 1), dtype=complex), k_space_trajectory])
    k_shifted_backward = np.vstack([k_space_trajectory, np.zeros((1, 1), dtype=complex)])
    grad_waveform = (k_shifted_forward - k_shifted_backward)[:-1] / sampling_period

    # Recalculate k-space positions at midpoints between time steps to match Pulseq definition
    initial_point = [grad_waveform[0] * sampling_period / 4]
    mid_points = (grad_waveform[:-1] + grad_waveform[1:]) * sampling_period / 2
    k_space_trajectory = -np.cumsum(np.concatenate((initial_point, mid_points)))

    # Compute final slew rate from the gradient waveform
    gradient_shifted_backward = np.vstack([np.zeros((1, 1), dtype=complex), grad_waveform])
    slew_rate = -np.diff(gradient_shifted_backward, axis=0) / sampling_period

    return (
        k_space_trajectory.flatten(),
        grad_waveform.flatten(),
        slew_rate.flatten(),
        time_points.flatten(),
        radial_positions.flatten(),
        angular_positions.flatten(),
    )
