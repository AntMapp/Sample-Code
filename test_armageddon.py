
from collections import OrderedDict
import pandas as pd
import numpy as np
import os

from pytest import fixture, mark

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly


@fixture(scope='module')
def armageddon():
    import armageddon
    return armageddon


@fixture(scope='module')
def planet(armageddon):
    return armageddon.Planet()


@fixture(scope='module')
def loc(armageddon):
    return armageddon.PostcodeLocator()


@fixture(scope='module')
def result(planet):
    input = {'radius': 1.,
             'velocity': 2.0e4,
             'density': 3000.,
             'strength': 1e5,
             'angle': 30.0,
             'init_altitude': 0.0,
             }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope='module')
def outcome(planet, result):
    result = planet.calculate_energy(result)
    outcome = planet.analyse_outcome(result=result)
    return outcome


def test_import(armageddon):
    assert armageddon


def test_planet_signature(armageddon):
    inputs = OrderedDict(atmos_func='exponential',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)

    # call by keyword
    planet = armageddon.Planet(**inputs)

    # call by position
    planet = armageddon.Planet(*inputs.values())  # noqa


def test_attributes(planet):
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)


def test_atmos_filename(planet):

    assert os.path.isfile(planet.atmos_filename)


def test_solve_atmospheric_entry(result):

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns


def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    assert type(energy) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns


def test_analyse_outcome(planet, outcome):

    assert type(outcome) is dict

    for key in ('outcome', 'burst_peak_dedz', 'burst_altitude',
                'burst_distance', 'burst_energy'):
        assert key in outcome.keys()


def test_damage_zones(armageddon):

    outcome = {'burst_peak_dedz': 1000.,
               'burst_altitude': 9000.,
               'burst_distance': 90000.,
               'burst_energy': 6000.,
               'outcome': 'Airburst'}

    blat, blon, damrad = armageddon.damage_zones(outcome, 55.0, 0., 135.,
                                                 [27e3, 43e3])

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 2


@mark.xfail
def test_great_circle_distance(armageddon):

    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])

    data = np.array([[1.28580537e+05, 2.59579735e+05, 2.25409117e+02],
                    [0.00000000e+00, 2.24656571e+05, 1.28581437e+05],
                    [2.72529953e+05, 2.08175028e+05, 1.96640630e+05]])

    dist = armageddon.great_circle_distance(pnts1, pnts2)

    assert np.allclose(data, dist, rtol=1.0e-4)


def test_locator_postcodes(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [0.2e3, 0.1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list


def test_locator_sectors(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [3.0e3, 1.5e3], True)

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list


def test_analyse_outcome_solution(planet):
    sol = {'outcome': 'Airburst',
           'burst_peak_dedz': 79.08758189042615,
           'burst_altitude': 33623.373340501596,
           'burst_distance': 187523.11473506835,
           'burst_energy': 356.86347496292706}

    res = planet.solve_atmospheric_entry(radius=10, angle=20,
                                         strength=1e6, density=3000,
                                         velocity=19e3)
    res = planet.calculate_energy(res)
    res = planet.analyse_outcome(res)

    assert res == sol


def test_damage_zones_numsol(armageddon):

    outcome_1 = {'burst_peak_dedz': 1000.,
                 'burst_altitude': 4e3,
                 'burst_distance': 90e3,
                 'burst_energy': 5e4,
                 'outcome': 'Airburst'}

    outcome_2 = {'burst_peak_dedz': 1000.,
                 'burst_altitude': 2e4,
                 'burst_distance': 90e3,
                 'burst_energy': 5e3,
                 'outcome': 'Airburst'}

    outcome_3 = {'burst_peak_dedz': 1000.,
                 'burst_altitude': 3.45e3,
                 'burst_distance': 90e3,
                 'burst_energy': 6.23e5,
                 'outcome': 'Airburst'}

    outcomes = [outcome_1, outcome_2, outcome_3]

    lat_arr = [52.79, 56.626614, 53.47653]
    lon_arr = [-2.95, -2.773774, -2.249616]

    bear_arr = [135, 110, 145]

    p_arr = [1657040.909608121,
             9452.853855227466,
             1254.8292399469913]

    r_arr = []
    dlat_arr = []
    dlon_arr = []

    r_sol_arr = [5e2, 3e2, 4.3e5]
    dlat_sol_arr = [52.21396905216966,
                    56.34218749238159,
                    52.811023577317435]
    dlon_sol_arr = [-2.015908861677074,
                    -1.4013791636652333,
                    -1.4815651079400531]

    for outcome, lat, lon, bear, pressure in zip(outcomes, lat_arr, lon_arr,
                                                 bear_arr, p_arr):
        dlat, dlon, damrad = armageddon.damage_zones(outcome, lat, lon, bear,
                                                     pressure)
        dlat_arr.append(dlat)
        dlon_arr.append(dlon)
        r_arr.append(damrad)

    assert np.allclose(r_sol_arr, r_arr, rtol=1.0e-4)
    assert np.allclose(dlat_sol_arr, dlat_arr, rtol=1.0e-4)
    assert np.allclose(dlon_sol_arr, dlon_arr, rtol=1.0e-4)


def test_postcodes_by_radius(loc):
    latlon = (51.4981, -0.1773)

    result1 = loc.get_postcodes_by_radius(latlon, [0.13e3])

    solution1 = [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD', 'SW7 5HF',
                  'SW7 5HG', 'SW7 5HQ']]

    result2 = loc.get_postcodes_by_radius(latlon, [0.13e3], True)

    solution2 = [['SW7 5', 'SW7 2']]

    assert sorted(result1[0]) == sorted(solution1[0])
    assert sorted(result2[0]) == sorted(solution2[0])


def test_population_of_postcode(loc):

    unit_post = [['AB101AB'], ['AB116FJ'], ['B13 0QD', 'B24 8AR']]
    ures = loc.get_population_of_postcode(unit_post)
    usol = [[0], [0], [60, 52]]

    sectoral_post = [['PE8 6', 'PE6 7'], ['PE9 3', 'PE9 4', 'PE9 2'],
                     ['PE9 1', 'LE157', 'NG335']]
    sres = loc.get_population_of_postcode(sectoral_post, True)
    ssol = [[8167, 9597], [6028, 6059, 9697], [9405, 8135, 5070]]

    assert ures == usol
    assert sres == ssol
