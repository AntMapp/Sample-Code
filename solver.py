import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(
            self,
            atmos_func='exponential',
            atmos_filename='./armageddon/resources/AltitudeDensityTable.csv',
            Cd=1.,
            Ch=0.1,
            Q=1e7,
            Cl=1e-3,
            alpha=0.3,
            Rp=6371e3,
            g=9.81,
            H=8000.,
            rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet
        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'
        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option
        Cd : float, optional
            The drag coefficient
        Ch : float, optional
            The heat transfer coefficient
        Q : float, optional
            The heat of ablation (J/kg)
        Cl : float, optional
            Lift coefficient
        alpha : float, optional
            Dispersion coefficient
        Rp : float, optional
            Planet radius (m)
        rho0 : float, optional
            Air density at zero altitude (kg/m^3)
        g : float, optional
            Surface gravity (m/s^2)
        H : float, optional
            Atmospheric scale height (m)
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        try:
            # set function to define atmoshperic density
            if atmos_func == 'exponential':
                self.rhoa = lambda z: rho0 * np.exp(-z / H)
            elif atmos_func == 'tabular':
                self.df_rhoa = pd.read_csv(
                    atmos_filename, sep=' ', skiprows=6, names=[
                        'Altitude', 'Density', 'Height'])
                self.rhoa = lambda z: self.df_rhoa.iloc[np.abs(
                    self.df_rhoa['Altitude'] - z).argmin()].Density

            elif atmos_func == 'constant':
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be \
                    'exponential', 'tabular' or 'constant'")

        except NotImplementedError:
            print("atmos_func {} not \
            implemented yet.".format(atmos_func))
            print("Falling back to constant \
            density atmosphere for now")
            self.rhoa = lambda x: rho0

    def ode_solver(self, t, dynamics, density, strength):
        """
        Solve the variable using each respective ODE equation as shown
        in the notebook
        Parameters
        ----------
        t : float
            The time
        dynamics : array of array
            The matrix to store ODE results for velocity, mass, angle,
            altitude, distance, and radius
        density : float
            The density of the asteroid in kg/m^3
        max_strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        Returns
        -------
        ode : array of array
            The matrix that contains the ODE results for velocity, mass,
            angle, altitude, distance, and radius
        """

        ode = np.zeros_like(dynamics)
        A = np.pi * dynamics[5]**2
        rhoa = self.rhoa(dynamics[3])

        ode[0] = -(self.Cd * rhoa * A * (dynamics[0])**2)\
            / (2 * dynamics[1]) + self.g * np.sin(dynamics[2])
        ode[1] = -(self.Ch * rhoa * A * (dynamics[0])**3)\
            / (2 * self.Q)
        ode[2] = (self.g * np.cos(dynamics[2])) / (dynamics[0])\
            - (self.Cl * rhoa * A * dynamics[0]) / (2 * dynamics[1])\
            - (dynamics[0] * np.cos(dynamics[2])) / (self.Rp + dynamics[3])
        ode[3] = -dynamics[0] * np.sin(dynamics[2])
        ode[4] = (dynamics[0] * np.cos(dynamics[2])) / \
            (1 + dynamics[3] / self.Rp)

        ode[5] = 0
        if rhoa * dynamics[0]**2 >= strength:
            ode[5] = ((7 / 2) * self.alpha * (rhoa / density)
                      )**(1 / 2) * dynamics[0]

        return ode

    def RK4(self, dt, y0, f, density, strength):
        """
        Solve the ODE using RK4 method
        Parameters
        ----------
        dt : float
            The time step
        y0 : array of array
            The matrix to store ODE results for velocity, mass, angle,
            altitude, distance, and radius
        f : function
            The function used to solve each respective ODE equation
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        Returns
        -------
        time_arr : array
            The array which stores each updating time

        y_arr : array of array
            The matrix that contains the ODE results for velocity, mass,
            angle, altitude, distance, and radius
        """
        y = np.array(y0)
        t = 0.

        y_arr = [y0]
        time_arr = [t]

        count = 0

        while y[3] > 0.1 and count < 1e4:
            k1 = dt * f(t, y, density, strength)
            k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1, density, strength)
            k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2, density, strength)
            k4 = dt * f(t + dt, y + k3, density, strength)
            y = y + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_arr.append(y)
            t += dt
            time_arr.append(t)

            count += 1

        return np.array(time_arr), np.array(y_arr)

    def forward_euler(self, dt, y0, f, density, strength):
        """
        Solve the ODE using Forward Euler method
        Parameters
        ----------
        dt : float
            The time step
        y0 : array of array
            The matrix to store ODE results for velocity, mass, angle,
            altitude, distance, and radius
        f : function
            The function used to solve each respective ODE equation
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        Returns
        -------
        time_arr : array
            The array which stores each updating time

        y_arr : array of array
            The matrix that contains the ODE results for velocity, mass,
            angle, altitude, distance, and radius
        """
        y = np.array(y0)
        t = 0
        y_arr = [y0]
        time_arr = [t]

        count = 0

        while y[3] > 0.1 and count < 1e4:
            y = y + dt * f(t, y, density, strength)  # euler guess
            y_arr.append(y)
            t = t + dt
            time_arr.append(t)

            count += 1

        return np.array(time_arr), np.array(y_arr)

    def improved_euler(self, dt, y0, f, density, strength):
        """
        Solve the ODE using Improved Euler method
        Parameters
        ----------
        dt : float
            The time step
        y0 : array of array
            The matrix to store ODE results for velocity, mass, angle,
            altitude, distance, and radius
        f : function
            The function used to solve each respective ODE equation
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        Returns
        -------
        time_arr : array
            The array which stores each updating time

        y_arr : array of array
            The matrix that contains the ODE results for velocity, mass,
            angle, altitude, distance, and radius
        """
        y = np.array(y0)
        t = 0.

        y_arr = [y0]
        time_arr = [0]

        count = 0

        while y[3] > 0.1 and count < 1e4:
            ye = y + dt * f(t, y, density, strength)  # euler guess
            y = y + 0.5 * dt * (f(t, y, density, strength) +
                                f(t + dt, ye, density, strength))
            y_arr.append(y)
            t = t + dt
            time_arr.append(t)

            count += 1

        return np.array(time_arr), np.array(y_arr)

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False, method='RK4'):
        """
        Solve the system of differential equations for a given impact scenario
        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters
        velocity : float
            The entery speed of the asteroid in meters/second
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians
        init_altitude : float, optional
            Initial altitude in m
        dt : float, optional
            The output timestep, in s
        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input
        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """

        if not radians:
            angle = np.radians(angle)

        mass = density * (4 * np.pi * radius**3) / 3
        distance = 0

        y0 = np.array([velocity, mass, angle, init_altitude, distance, radius])

        if method == 'RK4':
            time_arr, y_arr = self.RK4(
                dt, y0, self.ode_solver, density, strength)
        elif method == 'FE':
            time_arr, y_arr = self.forward_euler(
                dt, y0, self.ode_solver, density, strength)
        elif method == 'IE':
            time_arr, y_arr = self.improved_euler(
                dt, y0, self.ode_solver, density, strength)

        # angle = np.degrees(angle)

        return pd.DataFrame({'velocity': y_arr[:, 0],
                             'mass': y_arr[:, 1],
                             'angle': np.degrees(y_arr[:, 2]),
                             'altitude': y_arr[:, 3],
                             'distance': y_arr[:, 4],
                             'radius': y_arr[:, 5],
                             'time': time_arr}, index=range(len(time_arr)))

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.
        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = result.copy()

        e = ((1 / 2) * result['mass'] * result['velocity'] ** 2).diff(2)
        result['dedz'] = (e / (result['altitude'] / 1000).diff(2)) / (4.184e12)
        result['dedz'] = result['dedz'].shift(periods=-1)
        result['dedz'].iloc[0] = 0
        result['dedz'].iloc[-1] = 0

        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats
        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, \
            which should contain the key ``outcome`` (which should contain \
            one of the following strings: ``Airburst`` or ``Cratering``), \
            as well as the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, \
            ``burst_distance``, ``burst_energy``
        """

        # assign outcomes to corresponding result values
        burst_peak_dedz = result.dedz.max()
        b_idx = result.dedz.idxmax()

        burst_altitude = result.altitude[b_idx]
        burst_distance = result.distance[b_idx]

        # calculate kinetic energy, initial:
        # unit conversion: TNT to Joules
        E0 = 0.5 * result.mass[0] * result.velocity[0]**2 / 4.184e12

        # calculate burst energy
        # unit conversion: TNT to Joules

        if result.dedz.iloc[-1] == burst_peak_dedz:
            outcome = 'Cratering'
            E_residual = 0.5 * \
                result.mass.iloc[-1] * result.velocity.iloc[-1]**2 / 4.184e12
            E_burst = max(E_residual, E0 - E_residual)

        else:
            outcome = 'Airburst'
            E_burst = (
                E0 -
                0.5 *
                result.mass[b_idx] *
                result.velocity[b_idx]**2 /
                4.184e12)

        outcome = {'outcome': outcome,
                   'burst_peak_dedz': burst_peak_dedz,
                   'burst_altitude': burst_altitude,
                   'burst_distance': burst_distance,
                   'burst_energy': E_burst}

        return outcome

    def extensions2(self):
        '''
        Solve the parameters r, Y using given csv file

        '''
        data = pd.read_csv('./resources/ChelyabinskEnergyAltitude.csv')
        data.rename(
            columns={
                'Height (km)': 'altitude',
                'Energy Per Unit Length (kt Km^-1)': 'dedz'},
            inplace=True)
        data_peak_dedz = data['dedz'].max()
        max_index = data['dedz'].idxmax()
        data_burst_altitude = data.altitude[max_index]

        # firstly estimate the value of r (from google it is roughly 10 m)
        # Then estimate the value of Y (from google)
        Y = 1e6

        # Enter the given values
        v = 1.92e4
        rho = 3300
        theta = 18.3

        error = 1e2
        r_small = 0

        # r_array: Assumed r range
        # Y_array: Assumed Y range
        r_array = np.linspace(9, 11, 21)
        Y_array = np. linspace(1, 10, 10) * 1e6

        # run all assumed possible values
        for r in r_array:
            for Y in Y_array:
                result = self.solve_atmospheric_entry(
                    r, v, rho, Y, theta, init_altitude=100e3, dt=0.05,
                    radians=False, method='IE')
                result = self.calculate_energy(result)
                error_peak = abs(
                    (data_peak_dedz -
                     self.analyse_outcome(result)['burst_peak_dedz']) /
                    data_peak_dedz)
                error_z = abs(
                    (data_burst_altitude -
                     self.analyse_outcome(result)['burst_altitude'] /
                     1000) /
                    data_burst_altitude)
                if error > max(error_peak, error_z):
                    error = max(error_peak, error_z)
                    r_small = r
                    Y_small = Y
                # Store the values for returning

        return r_small, Y_small

    # Analytical solution
    # For special conditions, the ode condition can be simplified.

    def ode_simple(self, t, dynamics, density, strength):
        """
        Solve the variable using each respective ODE equation as shown
        in the notebook
        Parameters
        ----------
        t : float
            The time
        dynamics : array of array
            The matrix to store ODE results for velocity, mass, angle,
            altitude, distance, and radius
        density : float
            The density of the asteroid in kg/m^3
        max_strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        Returns
        -------
        ode : array of array
            The matrix that contains the ODE results for velocity, mass,
            angle, altitude, distance, and radius
        """

        ode = np.zeros_like(dynamics)
        A = np.pi * dynamics[5]**2
        rhoa = self.rhoa(dynamics[3])

        ode[0] = -(self.Cd * rhoa * A * (dynamics[0])**2)\
            / (2 * dynamics[1])
        ode[1] = 0
        ode[2] = 0
        ode[3] = -dynamics[0] * np.sin(dynamics[2])
        ode[4] = dynamics[0] * np.cos(dynamics[2])
        ode[5] = 0

        return ode

    def analytic_simple(self, radius, velocity, density, strength, angle,
                        init_altitude=100e3, error_tor=0.01, radians=False):
        '''
        Solve the simple condition analytic solution and compare the
        analytic v(z) with 3 different numerical methods(forward euler,
        improved euler and rk4) solutions in different time step length dt,
        dt = np.arange(0.001, 1., 0.005), and plot the maximum errors
        with respect to time step length dt.

        By the input error torlerance, derive the maximum time step rec_dt,
        which will narrow down the maximum relative error below
        error torlerance for the three numerical methods.

        By calculating and ploting the simple ODE system with time step rec_dt,
        can therefore derive the best fit numerical method for this analytic
        simple ODE system and can be used to solve the general ODE system.
        Parameters
        ----------

        radius : float

        The radius of the asteroid in meters

        velocity : float

        The entery speed of the asteroid in meters/second

        density : float

        The density of the asteroid in kg/m^3

        strength : float

        The strength of the asteroid (i.e. the maximum pressure it can

        take before fragmenting) in N/m^2

        angle : float

        The initial trajectory angle of the asteroid to the horizontal

        By default, input is in degrees. If 'radians' is set to True, the

        input should be in radians

        init_altitude : float, optional

        Initial altitude in m

        error_tor : float, optional

        The error torlerance, max relative error rate

        radians : logical, optional

        Whether angles should be given in degrees or radians. Default=False

        Angles returned in the dataframe will have the same units as the

        input

        Returns

        -------

        rec_dt : float

        The recommend maximum time step length dt

        '''

        if not radians:
            angle = np.radians(angle)
            radians = True

        # setting time steps
        dt_array = np.arange(0.001, 1, 0.005)

        mass = density * 4 / 3 * np.pi * radius**3

        # calculating analytical solution v(z)
        k = self.Cd * self.rho0 * np.pi * radius**2 * self.H / \
            (2 * mass * np.sin(angle))

        p = velocity * np.exp(k * np.exp(-1 * init_altitude / self.H))

        initial_distance = 0
        initial_condition = np.array(
            [velocity, mass, angle, init_altitude, initial_distance, radius])

        # initialize error
        error_velocity_rk4 = []
        error_velocity_forward = []
        error_velocity_improved = []

        # calculating error for different time steps
        for dt in dt_array:

            # rk4
            time_arr_rk4, result_arr_rk4 = self.RK4(
                dt, initial_condition, self.ode_simple, density, strength)
            result_arr_rk4 = np.array(result_arr_rk4)
            result_arr_rk4 = result_arr_rk4[:-1, ]
            time_arr_rk4 = np.array(time_arr_rk4)
            time_arr_rk4 = time_arr_rk4[:-1, ]

            data_rk4 = pd.DataFrame(
                {'velocity': result_arr_rk4[:, 0],
                 'mass': result_arr_rk4[:, 1],
                 'angle': np.degrees(result_arr_rk4[:, 2]),
                 'altitude': result_arr_rk4[:, 3],
                 'distance': result_arr_rk4[:, 4],
                 'radius': result_arr_rk4[:, 5],
                 'time': time_arr_rk4}, index=range(len(time_arr_rk4)))

            v_z_analytic_rk4 = p * \
                np.exp(-k * np.exp(-1 * data_rk4['altitude'] / self.H))
            data_rk4['v_z_analytic'] = v_z_analytic_rk4

            data_rk4 = self.calculate_energy(data_rk4)
            dedz_analytic_rk4 = (0.5 * mass * v_z_analytic_rk4**2).diff() / \
                ((data_rk4['altitude'] / 1000).diff() * (4.184e12))

            data_rk4['dedz_analytic_rk4'] = dedz_analytic_rk4
            data_rk4['dedz_analytic_rk4'].iloc[0] = 0
            data_rk4['error'] = abs(
                data_rk4['dedz'] - data_rk4['dedz_analytic_rk4']) \
                / data_rk4['dedz_analytic_rk4']

            data_rk4['error_rk4'] = abs(
                data_rk4['velocity'] - data_rk4['v_z_analytic']) \
                / data_rk4['v_z_analytic']

            error_velocity_rk4.append(data_rk4['error_rk4'].max())

            # forwawrd euler
            time_arr_forward, result_arr_forward = self.forward_euler(
                dt, initial_condition, self.ode_simple, density, strength)
            result_arr_forward = np.array(result_arr_forward)
            result_arr_forward = result_arr_forward[:-1, ]
            time_arr_forward = np.array(time_arr_forward)
            time_arr_forward = time_arr_forward[:-1, ]

            data_forward = pd.DataFrame(
                {'velocity': result_arr_forward[:, 0],
                 'mass': result_arr_forward[:, 1],
                 'angle': np.degrees(result_arr_forward[:, 2]),
                 'altitude': result_arr_forward[:, 3],
                 'distance': result_arr_forward[:, 4],
                 'radius': result_arr_forward[:, 5],
                 'time': time_arr_forward}, index=range(len(time_arr_forward)))

            v_z_analytic_forward = p * \
                np.exp(-k * np.exp(-1 * data_forward['altitude'] / self.H))
            data_forward['v_z_analytic'] = v_z_analytic_forward

            data_forward = self.calculate_energy(data_forward)
            dedz_analytic_forward = (
                0.5 * mass * v_z_analytic_forward**2).diff() / (
                (data_forward['altitude'] / 1000).diff() * (4.184e12))

            data_forward['dedz_analytic_forward'] = dedz_analytic_forward

            data_forward['dedz_analytic_forward'].iloc[0] = 0
            data_forward['error'] = abs(
                data_forward['dedz'] - data_forward['dedz_analytic_forward']) \
                / data_forward['dedz_analytic_forward']

            data_forward['error_forward'] = abs(
                data_forward['velocity'] - data_forward['v_z_analytic']) \
                / data_forward['v_z_analytic']

            error_velocity_forward.append(data_forward['error_forward'].max())

            # improved euler
            time_arr_improved, result_arr_improved = self.improved_euler(
                dt, initial_condition, self.ode_simple, density, strength)
            result_arr_improved = np.array(result_arr_improved)
            result_arr_improved = result_arr_improved[:-1, ]
            time_arr_improved = np.array(time_arr_improved)
            time_arr_improved = time_arr_improved[:-1, ]

            data_improved = pd.DataFrame(
                {'velocity': result_arr_improved[:, 0],
                 'mass': result_arr_improved[:, 1],
                 'angle': np.degrees(result_arr_improved[:, 2]),
                 'altitude': result_arr_improved[:, 3],
                 'distance': result_arr_improved[:, 4],
                 'radius': result_arr_improved[:, 5],
                 'time': time_arr_improved},
                index=range(len(time_arr_improved)))

            v_z_analytic_improved = p * \
                np.exp(-k * np.exp(-1 * data_improved['altitude'] / self.H))
            data_improved['v_z_analytic'] = v_z_analytic_improved

            data_improved = self.calculate_energy(data_improved)
            dedz_analytic_improved = (
                0.5 * mass * v_z_analytic_improved**2).diff() / (
                (data_improved['altitude'] / 1000).diff() * (4.184e12))

            data_improved['dedz_analytic_improved'] = dedz_analytic_improved
            data_improved['dedz_analytic_improved'].iloc[0] = 0
            data_improved['error'] = abs(
                data_improved['dedz']
                - data_improved['dedz_analytic_improved'])\
                / data_improved['dedz_analytic_improved']

            data_improved['error_improved'] = abs(
                data_improved['velocity']
                - data_improved['v_z_analytic']) / \
                data_improved['v_z_analytic']

            error_velocity_improved.append(
                data_improved['error_improved'].max())

        # find the largerest dt which narrows the maximum
        # error below error_tor
        rec_dt = 0

        steps = np.ones(200)

        for i in range(len(dt_array)):

            if error_velocity_forward[i] >= error_tor:

                rec_dt = dt_array[i - 1]

                break

        # using three different numercial methods to derive the results
        # and plot them together in a graph
        # rk4
        time_arr_rk4, result_arr_rk4 = self.RK4(
            rec_dt, initial_condition, self.ode_simple, density, strength)
        result_arr_rk4 = np.array(result_arr_rk4)
        result_arr_rk4 = result_arr_rk4[:-1, ]
        time_arr_rk4 = np.array(time_arr_rk4)
        time_arr_rk4 = time_arr_rk4[:-1, ]

        data_rk4 = pd.DataFrame(
            {'velocity': result_arr_rk4[:, 0],
             'mass': result_arr_rk4[:, 1],
             'angle': np.degrees(result_arr_rk4[:, 2]),
             'altitude': result_arr_rk4[:, 3],
             'distance': result_arr_rk4[:, 4],
             'radius': result_arr_rk4[:, 5],
             'time': time_arr_rk4},
            index=range(len(time_arr_rk4)))

        v_z_analytic_rk4 = p * \
            np.exp(-k * np.exp(-1 * data_rk4['altitude'] / self.H))
        data_rk4['v_z_analytic'] = v_z_analytic_rk4

        data_rk4 = self.calculate_energy(data_rk4)
        dedz_analytic_rk4 = (0.5 * mass * v_z_analytic_rk4**2).diff() / \
            ((data_rk4['altitude'] / 1000).diff() * (4.184e12))

        data_rk4['dedz_analytic_rk4'] = dedz_analytic_rk4
        data_rk4['dedz_analytic_rk4'].iloc[0] = 0
        data_rk4['error'] = abs(
            data_rk4['dedz'] - data_rk4['dedz_analytic_rk4']) \
            / data_rk4['dedz_analytic_rk4']

        data_rk4['error_rk4'] = abs(
            data_rk4['velocity'] - data_rk4['v_z_analytic']) \
            / data_rk4['v_z_analytic']

        # forwawrd euler
        time_arr_forward, result_arr_forward = self.forward_euler(
            rec_dt, initial_condition, self.ode_simple, density, strength)
        result_arr_forward = np.array(result_arr_forward)
        result_arr_forward = result_arr_forward[:-1, ]
        time_arr_forward = np.array(time_arr_forward)
        time_arr_forward = time_arr_forward[:-1, ]

        data_forward = pd.DataFrame(
            {'velocity': result_arr_forward[:, 0],
             'mass': result_arr_forward[:, 1],
             'angle': np.degrees(result_arr_forward[:, 2]),
             'altitude': result_arr_forward[:, 3],
             'distance': result_arr_forward[:, 4],
             'radius': result_arr_forward[:, 5],
             'time': time_arr_forward},
            index=range(len(time_arr_forward)))

        v_z_analytic_forward = p * \
            np.exp(-k * np.exp(-1 * data_forward['altitude'] / self.H))
        data_forward['v_z_analytic'] = v_z_analytic_forward

        data_forward = self.calculate_energy(data_forward)
        dedz_analytic_forward = (
            0.5 * mass * v_z_analytic_forward**2).diff() / (
            (data_forward['altitude'] / 1000).diff() * (4.184e12))

        data_forward['dedz_analytic_forward'] = dedz_analytic_forward

        data_forward['dedz_analytic_forward'].iloc[0] = 0
        data_forward['error'] = abs(
            data_forward['dedz'] - data_forward['dedz_analytic_forward']) \
            / data_forward['dedz_analytic_forward']

        data_forward['error_forward'] = abs(
            data_forward['velocity'] - data_forward['v_z_analytic']) \
            / data_forward['v_z_analytic']

        # improved euler
        time_arr_improved, result_arr_improved = self.improved_euler(
            rec_dt, initial_condition, self.ode_simple, density, strength)
        result_arr_improved = np.array(result_arr_improved)
        result_arr_improved = result_arr_improved[:-1, ]
        time_arr_improved = np.array(time_arr_improved)
        time_arr_improved = time_arr_improved[:-1, ]

        data_improved = pd.DataFrame(
            {'velocity': result_arr_improved[:, 0],
             'mass': result_arr_improved[:, 1],
             'angle': np.degrees(result_arr_improved[:, 2]),
             'altitude': result_arr_improved[:, 3],
             'distance': result_arr_improved[:, 4],
             'radius': result_arr_improved[:, 5],
             'time': time_arr_improved},
            index=range(len(time_arr_improved)))

        v_z_analytic_improved = p * \
            np.exp(-k * np.exp(-1 * data_improved['altitude'] / self.H))
        data_improved['v_z_analytic'] = v_z_analytic_improved

        data_improved = self.calculate_energy(data_improved)
        dedz_analytic_improved = (
            0.5 * mass * v_z_analytic_improved**2).diff() / (
            (data_improved['altitude'] / 1000).diff() * (4.184e12))

        data_improved['dedz_analytic_improved'] = dedz_analytic_improved
        data_improved['dedz_analytic_improved'].iloc[0] = 0
        data_improved['error'] = abs(
            data_improved['dedz'] - data_improved['dedz_analytic_improved']) \
            / data_improved['dedz_analytic_improved']

        data_improved['error_improved'] = abs(
            data_improved['velocity'] - data_improved['v_z_analytic']) \
            / data_improved['v_z_analytic']

        # recommanded scheme to implement
        error_mean_rk4 = data_rk4['error_rk4'].mean()
        error_mean_forward = data_forward['error_forward'].mean()
        error_mean_improved = data_improved['error_improved'].mean()

        if min(
                error_mean_rk4,
                error_mean_forward,
                error_mean_improved) == error_mean_forward:

            print(
                f'Based on the velocity error, \
                recommend maximum step length dt is {rec_dt}, \
                error_torlerance is set to be {error_tor}\n',
                f'For this time step, the maximum error for forward euler, \
                improved euler and rk4 is below \
                error_torlerance = {error_tor}\n',
                'the recommond scheme to implement is rk4.\n')

        elif min(error_mean_rk4, error_mean_forward,
                 error_mean_improved) == error_mean_improved:

            print(
                f'Based on the velocity error, \
                recommend maximum step length dt is {rec_dt},\
                 error_torlerance is set to be {error_tor}\n',
                f'For this time step, the maximum error for forward euler, \
                improved euler and rk4 is below \
                error_torlerance = {error_tor}\n',
                'the recommond scheme to implement is rk4.\n')

        else:

            print(
                f'Based on the velocity error, recommend \
                 maximum step length dt is {rec_dt},\
                 error_torlerance is set to be {error_tor}\n',
                f'For this time step, the maximum error for forward euler,\
                 improved euler and rk4 is \
                 below error_torlerance = {error_tor}\n',
                'the recommond scheme to implement is rk4.\n')

        # errors visulization
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.reshape(-1)

        axs[1].plot(
            np.asarray(
                data_rk4['altitude']),
            np.asarray(
                data_rk4['error_rk4']),
            'b-',
            label='velocity relative error of rk4 method')
        axs[1].plot(
            np.asarray(
                data_forward['altitude']),
            np.asarray(
                data_forward['error_forward']),
            'r-',
            label='velocity relative error of forward euler method')
        axs[1].plot(
            np.asarray(
                data_improved['altitude']),
            np.asarray(
                data_improved['error_improved']),
            'g-',
            label='velocity relative error of improved forward euler method')

        axs[1].set_ylabel('$Error$', fontsize=7)
        axs[1].set_xlabel('$altitude (m)$', fontsize=7)
        axs[1].set_title(
            f'velocity error with respect to altitude, \
            error torlerance = {error_tor}',
            fontsize=7)
        axs[1].legend(loc='best', fontsize=7)

        # errors visulization
        axs[0].plot(
            np.asarray(dt_array),
            np.asarray(error_velocity_rk4),
            'b-',
            label='error in velocity, rk4')
        axs[0].plot(
            np.asarray(dt_array),
            np.asarray(error_velocity_forward),
            'r-',
            label='error in velocity, forward euler')
        axs[0].plot(
            np.asarray(dt_array),
            np.asarray(error_velocity_improved),
            'g-',
            label='error in velocity, improved euler')
        axs[0].plot(np.asarray(dt_array), error_tor *
                    steps, 'y-', label=f'error torlerance = {error_tor}')
        axs[0].set_xlabel('dt', fontsize=10)
        axs[0].set_ylabel('velocity error', fontsize=10)
        axs[0].set_title('error in velocity', fontsize=10)
        axs[0].legend(loc='best', fontsize=7)

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.22, hspace=0.35)
        plt.show()

        return rec_dt
