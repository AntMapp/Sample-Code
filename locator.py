"""Module dealing with postcode information."""

import numpy as np
import pandas as pd


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)}
    >>> with numpy.printoptions(formatter={'all', fmt}):
        print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [1.286e+05 6.378e+04]
    """
    distance = np.empty((len(latlon1), len(latlon2)), float)

    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)

    Rp = 6371 * 1000

    if (latlon1.ndim == 1):
        latlon1 = np.copy([latlon1])
    else:
        latlon1 = np.copy(latlon1)
    if (latlon2.ndim == 1):
        latlon2 = np.copy([latlon2])
    else:
        latlon2 = np.copy(latlon2)

    latlon1 = latlon1 * np.pi / 180
    latlon2 = latlon2 * np.pi / 180
    # Calculate distance
    distance = Rp * 2 * np.arcsin(np.sqrt((np.sin(np.absolute(
        np.subtract.outer(latlon1[:, 0], latlon2[:, 0])) / 2))**2 +
        np.multiply.outer(np.cos(latlon1[:, 0]), np.cos(latlon2[:, 0])) *
        (np.sin(np.absolute(np.subtract.outer(
                latlon1[:, 1], latlon2[:, 1]))) / 2)**2))

    return distance


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""
    postcode_file = './armageddon/resources/data/full_postcodes.csv'
    census_file = \
        './armageddon/resources/data/population_by_postcode_sector.csv'

    def __init__(self, postcode_file=postcode_file,
                 census_file=census_file,
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in latitude
            -longitude space.

        """
        self.norm = norm

        # Read csv file
        self.postcode_file = pd.read_csv(postcode_file, header=0)
        self.census_file = pd.read_csv(census_file, header=0)

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Return (unit or sector) postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements of
            radii to the location X.


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD',\
         'SW7 5HF', 'SW7 5HG', 'SW7 5HQ']]
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773),\
        >>> [0.4e3, 0.2e3], True)
        [['SW7 5', 'SW7 9', 'SW7 4', 'SW7 3', 'SW7 1', 'SW7 2'],
         ['SW7 5', 'SW7 9', 'SW7 4', 'SW7 3', 'SW7 1', 'SW7 2']]

        """
        df = self.postcode_file.copy()
        Rp = 6371 * 1000
        list_list_postcodes = []
        max_r = max(radii)
        # Add degree bias to the four directions
        bias = 1e-2 * 180 / np.pi

        max_degree = max_r * 180 / np.pi

        normal1 = max_degree / Rp

        if normal1 >= 0:
            lamax = X[0] + normal1 + bias
            lamin = X[0] - normal1 - bias
        else:
            lamax = X[1] - normal1 + bias
            lamin = X[1] + normal1 - bias

        normal2 = max_degree / (Rp * np.cos(X[0]))
        if normal2 >= 0:
            lomax = X[1] + normal2 + bias
            lomin = X[1] - normal2 - bias
        else:
            lomax = X[1] - normal2 + bias
            lomin = X[1] + normal2 - bias

        # Filter to a square space
        df_select = df[(df['Latitude'] <= lamax) &
                       (df['Latitude'] >= lamin) &
                       (df['Longitude'] <= lomax) &
                       (df['Longitude'] >= lomin)]

        postcodes = df_select['Postcode'].values

        near_postcode_latlon = df_select.loc[:, ['Latitude', 'Longitude']]

        if near_postcode_latlon.empty:
            return None

        distance_list = self.norm(X, near_postcode_latlon).flatten()

        for r in radii:
            units = list(postcodes[distance_list < r])
            if sector:
                sectors = [unit[0:5] for unit in units]
                sectors = list(set(sectors))
                list_list_postcodes.append(sectors)
            else:
                list_list_postcodes.append(units)
        return list_list_postcodes

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors,
            otherwise postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator()
        >>> locator.get_population_of_postcode([['SW7 2AZ','SW7 2BT',
                                                 'SW7 2BU','SW7 2DD']])

        >>> [[19, 19, 19, 19]]

        >>> locator.get_population_of_postcode([['SW7  2']], True)
        >>> [[2283]]
        """
        p_f = self.postcode_file.copy()
        p_f.Postcode = p_f.Postcode.fillna(0)
        total_population = 'Variable: All usual residents; measures: Value'

        # If sectoral postcode
        if sector:
            postcode_lst = [i[:-1] + ' ' + i[-1:] if len(i) != 6 else i
                            for item in postcodes for i in item]

            # Select population data
            population = self.census_file[
                self.census_file['geography']
                .isin(postcode_lst)][['geography',
                                     total_population]]\
                .set_index('geography')

            # Restore ordering
            population = population.reindex(postcode_lst, fill_value=0)[
                total_population]\
                .values.tolist()

        # Else, inputs are unit postcode
        else:
            p_f = self.postcode_file.copy()
        p_f.Postcode = p_f.Postcode.fillna(0)
        total_population = 'Variable: All usual residents; measures: Value'

        # If sectoral postcode
        if sector:
            postcode_lst = [i[:-1] + ' ' + i[-1:] if len(i) != 6 else i
                            for item in postcodes for i in item]

            # Select population data
            population = self.census_file[
                self.census_file['geography']
                .isin(postcode_lst)][['geography',
                                     total_population]]\
                .set_index('geography')

            # Restore ordering
            population = population.reindex(postcode_lst, fill_value=0)[
                total_population]\
                .values.tolist()

        # Else, inputs are unit postcode
        else:
            postcode_lst = []

            for i in range(len(postcodes)):
                postcode_lst = np.append(postcode_lst, postcodes[i])

            postcode_s = pd.Series(postcode_lst,
                                   index=range(len(postcode_lst)))
            postcode_s = postcode_s.str.slice(stop=5)
            population_lst = postcode_s
            postcode_s = postcode_s.unique()

            for postcode in postcode_s:
                postcode_r = postcode[:4] + ' ' + postcode[4]

                if (len(self.census_file[
                    self.census_file.geography == postcode_r]
                        [total_population])) == 0:
                    avg_population = 0

                else:
                    num_same_post = p_f.Postcode.str.\
                                    startswith(postcode).sum()
                    avg_population = int(self.census_file[
                                         self.census_file.geography ==
                                         postcode_r]
                                         [total_population].item()
                                         / num_same_post)

                population_lst = population_lst.replace(
                                {postcode: avg_population})
            population = population_lst.tolist()

        # Restore original shape
        res_array = []
        start = 0
        for i in range(len(postcodes)):
            end = start + len(postcodes[i])
            if len(population[start:end]) == 0:
                res_array.append([0])
            else:
                res_array.append(population[start:end])
            start = end

        return res_array
