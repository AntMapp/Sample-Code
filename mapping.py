import folium


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle([lat, lon], radius, fill=True,
                  fillOpacity=0.6, **kwargs).add_to(map)

    return map

# damage_map = plot_circle(52.79, -2.95, 1e3, map=None)
# damage_map.save("damage_map.html")


def plot_circles(lat, lon, dlat, dlon, radius, map=None, **kwargs):
    """
    Plot concentric circles on a map (creating a new folium map instance if
    necessary).

    Parameters
    ----------

    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    dlat: float
        latitude of circle to plot (degrees)
    dlon: float
        longitude of circle to plot (degrees)
    radius: list
        radius of circles to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, [1e3, 2e6], map=None)
    """

    if not map:
        map = folium.Map(location=[dlat, dlon], control_scale=True)
    for r in radius:
        if r == min(radius):
            folium.Circle([dlat, dlon], r, fill=True, fillOpacity=0.6,
                          fill_color="red", **kwargs).add_to(map)
        else:
            folium.Circle([dlat, dlon], r, fill=True, fillOpacity=0.6,
                          **kwargs).add_to(map)

    folium.PolyLine(
        locations=[
            [lat, lon],
            [dlat, dlon],
        ],
        color='black'
    ).add_to(map)

    return map
