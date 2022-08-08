import numpy as np
import pandas as pd

from scipy.optimize import fsolve

from armageddon.solver import Planet
from armageddon.locator import PostcodeLocator

# from solver import Planet
# from locator import PostcodeLocator
from armageddon import mapping

import folium
import webbrowser


def locate(outcome, lat, lon, bearing):
    """
    Calculate the latitude and longitude of the surface zero location.
    Parameters
    ----------
    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    Returns
    -------
    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    Examples
    --------
    >>> import armageddon
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> armageddon.locate(outcome, 52.79, -2.95, 135)
    """
    # radius of the earth
    Rp = 6.371e6
    # temporary value representing r/Rp, \r: burst distance
    temp = outcome['burst_distance']/Rp
    # convert to radian
    lat, lon, bearing = np.deg2rad([lat, lon, bearing])
    blat = np.arcsin(np.sin(lat)*np.cos(temp) +
                     np.cos(lat)*np.sin(temp)*np.cos(bearing))
    blon = lon + np.arctan(np.sin(bearing) * np.sin(temp) * np.cos(lat) /
                           (np.cos(temp)-np.sin(lat) * np.sin(blat)))
    # convert to degree
    blat, blon = np.rad2deg([blat, blon])
    return blat, blon


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.
    Parameters
    ----------
    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels
    Returns
    -------
    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii for the input damage\
        levels
    Examples
    --------
    >>> import armageddon
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,
                   'burst_distance': 90e3, 'burst_peak_dedz': 1e3,
                   'outcome': 'Airburst'}
    >>> armageddon.damage_zones(outcome, 52.79, -2.95, 135, pressures=[1e3,\
        3.5e3, 27e3, 43e3])
    """
    # Replace this code with your own. For demonstration we return lat,
    # lon and 1000 m
    # blat = lat
    # blon = lon
    # damrad = [5000.] * len(pressures)
    if type(pressures) == float:
        pressures = [pressures]

    blat, blon = locate(outcome, lat, lon, bearing)

    # Function to generate equations for solving r values using Scipy package
    def generateE(r, zb=np.copy(outcome['burst_altitude']),
                  ek=np.copy(outcome['burst_energy'])):
        equations = [0.]*len(pressures)

        # iterate through pressures list and form equations relatively
        for i, p in enumerate(pressures):
            temp = r[i] ** 2 + zb ** 2
            equations[i] = 3.14e11 * (temp / ek ** (2 / 3)) ** (-1.3)\
                + 1.8e7 * (temp / ek ** (2 / 3)) ** (-0.565) - p
        return equations

    # Call fsolve function to solve equations
    sol_arr = fsolve(generateE, [0.] * len(pressures))
    sol_list = sol_arr.tolist()

    for i, sol in enumerate(sol_list):
        sol_list[i] = np.absolute(sol)

    if len(sol_list) == 1:
        damrad = sol_list[0]
    else:
        damrad = sol_list

    return float(blat), float(blon), damrad


fiducial_means = {'radius': 10, 'angle': 20, 'strength': 1e6,
                  'density': 3000, 'velocity': 19e3,
                  'lat': 51.5, 'lon': 1.5, 'bearing': -45.}
fiducial_stdevs = {'radius': 1, 'angle': 1, 'strength': 5e5,
                   'density': 500, 'velocity': 1e3,
                   'lat': 0.025, 'lon': 0.025, 'bearing': 0.5}


def impact_risk(planet, means=fiducial_means, stdevs=fiducial_stdevs,
                pressure=27.e3, nsamples=100, sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector
    Parameters
    ----------
    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry
    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``
    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``
    pressure: float
        The pressure at which to calculate the damage zone for each impact
    nsamples: int
        The number of iterations to perform in the uncertainty analysis
    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes
    Returns
    -------
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    """
    # Randomly generate parameters
    np.random.seed(0)
    random_para = {}
    for key in means:
        random_para[key] = np.random.normal(means[key],
                                            stdevs[key], nsamples)
    planet = Planet()

    all_postcodes = []
    postcode_locator = PostcodeLocator()
    # Define a map
    m = folium.Map(
        location=[51.4981, -0.1773],
        control_scale=True,
        zoom_start=7
    )

    for i in range(nsamples):
        random_radius = random_para['radius'][i]
        random_angle = random_para['angle'][i]
        random_strength = random_para['strength'][i]
        random_density = random_para['density'][i]
        random_velocity = random_para['velocity'][i]
        random_lat = random_para['lat'][i]
        random_lon = random_para['lon'][i]
        random_bearing = random_para['bearing'][i]

        # print("random", random_radius)
        # Solving ODE
        temp = planet.solve_atmospheric_entry(radius=random_radius,
                                              velocity=random_velocity,
                                              density=random_density,
                                              strength=random_strength,
                                              angle=random_angle)

        temp1 = planet.calculate_energy(temp)
        outcome = planet.analyse_outcome(temp1)

        # Calculate damage zones
        lat, lon, rad = damage_zones(outcome, random_lat, random_lon,
                                     random_bearing,
                                     pressures=[pressure])
        # Add circle to the map
        mapping.plot_circle(lat, lon, rad, map=m)

        # Get postcodes
        if not isinstance(rad, list):
            postcode = postcode_locator.get_postcodes_by_radius(X=(lat, lon),
                                                                radii=[rad],
                                                                sector=sector)
        else:
            postcode = postcode_locator.get_postcodes_by_radius(X=(lat, lon),
                                                                radii=rad,
                                                                sector=sector)
        if not postcode:
            continue
        all_postcodes.extend(postcode[0])

    m.save('risk_map.html')
    webbrowser.open('risk_map.html')
    # Count the number of each postcode
    count_dict = {}
    for key in all_postcodes:
        count_dict[key] = count_dict.get(key, 0) + 1
    # print(count_dict)
    # Calculate the possiblity
    possible_dict = {}
    for key in count_dict:
        possible_dict[key] = count_dict[key] / nsamples
    # print(possible_dict)

    postcode_list = list(possible_dict.keys())
    # print(postcode_list)
    # Get the population
    popula_list = postcode_locator.get_population_of_postcode([postcode_list],
                                                              sector)
    popula_list = popula_list[0]
    # print(popula_list)
    risk_dict = {}
    # Calculate risk
    for i, key in enumerate(possible_dict):
        risk_dict[key] = possible_dict[key] * popula_list[i]
    # data = {'Risk': risk_dict}
    # df = pd.DataFrame(data)

    unit_sector_list = list(risk_dict.keys())
    risk_list = list(risk_dict.values())

    if sector:
        return pd.DataFrame({'sector': unit_sector_list, 'risk': risk_list})
    else:
        return pd.DataFrame({'postcode': unit_sector_list, 'risk': risk_list})
