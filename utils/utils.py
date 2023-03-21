import matplotlib.pyplot as plt
from ipywidgets import HTML
import ipyleaflet as ipy


def plot_boxplots_modes(tl, column_name, lim, ylabel=None):
    """Visualise box plots for every mode"""

    tl["mode"] = tl["mode"].replace(
        {
            "Mode::Car": "Car",
            "Mode::Ecar": "Car",
            "Mode::Bicycle": "Bicycle",
            "Mode::Ebicycle": "Bicycle",
            "Mode::Walk": "Walk",
            "Mode::Train": "Train",
            "Mode::Tram": "Tram",
            "Mode::Bus": "Bus",
            "Mode::Boat": "Boat",
        }
    )

    data = (
        tl[tl["mode"] == "Bicycle"][column_name],
        tl[tl["mode"] == "Boat"][column_name],
        tl[tl["mode"] == "Bus"][column_name],
        tl[tl["mode"] == "Car"][column_name],
        tl[tl["mode"] == "Train"][column_name],
        tl[tl["mode"] == "Tram"][column_name],
        tl[tl["mode"] == "Walk"][column_name],
        tl[column_name],
    )

    plt.figure(figsize=(12, 8))
    plt.boxplot(
        data,
        positions=range(0, 16, 2),
        widths=1.5,
        patch_artist=True,
        flierprops={"markeredgecolor": "palevioletred", "markersize": 4},
        medianprops={"color": "white", "linewidth": 0.5},
        boxprops={"facecolor": "palevioletred", "edgecolor": "white", "linewidth": 0.5},
        whiskerprops={"color": "palevioletred", "linewidth": 1.5},
        capprops={"color": "palevioletred", "linewidth": 1.5},
    )
    plt.xticks(range(0, 16, 2), ["bicycle", "boat", "bus", "car", "train", "tram", "walk", "TOTAL"])
    plt.ylim(lim[0], lim[1])
    plt.ylabel(ylabel)
    plt.grid()
    plt.title("Boxplots of feature '" + column_name + "'", fontsize=14)
    plt.show()


def plot_hist(tl, column_name, bin_size, xlabel=None, ylabel=None):
    """Visualize customised histogram"""
    data = tl[column_name]
    minima = min(data)
    maxima = max(data)
    bins = range(int(minima), int(maxima) + 2, bin_size)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color="palevioletred")
    plt.title(column_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(minima, maxima + 2)
    plt.grid()
    plt.show()


def new_map():
    """Create new leaflet map"""
    m = ipy.Map(center=(46.8131873, 8.22421), zoom=8, basemap=ipy.basemaps.OpenStreetMap.Mapnik)
    m.add_control(ipy.LayersControl())
    return m


def draw_line(coords, color, message):
    """Draw line on leaflet map"""
    coord_list = [[coord[1], coord[0]] for coord in coords]
    line = ipy.Polyline()
    line.color = color
    line.locations = coord_list
    line.weight = 2
    line.fill = False
    if message:
        line.popup = message
    return line


def show_geometries(m, df, geom_column="geometry", info_column=False, color="blue", layer_name="object"):
    """Draw point, line or polygon on leaflet map"""
    objects = list()

    for ix, row in df.iterrows():
        if info_column:
            message = HTML()
            message.value = """<table>
                                    <tr> <td>index:</td>      <td>&emsp;</td> <td>{}</td> </tr>
                                    <tr> <td>{}</td>      <td>&emsp;</td> <td>{}</td> </tr>
                               </table>""".format(
                ix, info_column, row[info_column]
            )
        else:
            message = False

        geom = row[geom_column]

        if geom.geom_type == "LineString":
            line = draw_line(geom.coords, color, message)
            objects.append(line)

        elif geom.geom_type == "MultiLineString":
            for element in geom.geoms:
                line = draw_line(element.coords, color, message)
                objects.append(line)

        elif geom.geom_type == "Polygon":
            line = draw_line(geom.exterior.coords, color, message)
            objects.append(line)
            for poly_in in geom.interiors:
                line = draw_line(poly_in.coords, color, message)
                objects.append(line)

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                line = draw_line(poly.exterior.coords, color, message)
                objects.append(line)
                for poly_in in poly.interiors:
                    line = draw_line(poly_in.coords, color, message)
                    objects.append(line)

        else:
            raise AttributeError(
                f"geom_type unknown. We only support ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']. You passed {geom.geom_type}"
            )

    layer_object = ipy.LayerGroup(layers=objects)
    layer_object.name = layer_name

    m.add_layer(layer_object)


def show_trajectories(m, df, show_vertices=False, color="red", layer_name="trajectories"):
    """Draw trajectories with or without track points on leaflet map"""

    trajectories = list()
    for ix, row in df.iterrows():
        message = HTML()
        message.value = """<table>
                               <tr> <td>index:</td>      <td>&emsp;</td> <td>{}</td> </tr>
                               <tr> <td>mode:</td>  <td>&emsp;</td> <td>{}</td> </tr>
                           </table>""".format(
            ix, row["mode"]
        )

        coord_list = [[coords[1], coords[0]] for coords in row["geom"].coords]

        if show_vertices:
            for coord in coord_list:
                circle = ipy.CircleMarker()
                circle.location = coord
                circle.radius = 5
                circle.fill_color = "blue"
                circle.fill_opacity = 1
                circle.stroke = False
                trajectories.append(circle)

        line = ipy.Polyline()
        line.color = color
        line.locations = coord_list
        line.weight = 2
        line.fill = False
        line.popup = message
        trajectories.append(line)

    layer_trajectories = ipy.LayerGroup(layers=trajectories)
    layer_trajectories.name = layer_name

    m.add_layer(layer_trajectories)


def show_trajectories_modes(m, df):
    """Draw trajectories separated by mode on leaflet map"""
    show_trajectories(m, df[df["mode"] == "Mode::Train"], show_vertices=False, color="Orange", layer_name="train")
    show_trajectories(m, df[df["mode"] == "Mode::Car"], show_vertices=False, color="RoyalBlue", layer_name="car")
    show_trajectories(m, df[df["mode"] == "Mode::Bus"], show_vertices=False, color="Red", layer_name="bus")
    show_trajectories(m, df[df["mode"] == "Mode::Tram"], show_vertices=False, color="YellowGreen", layer_name="tram")
    show_trajectories(m, df[df["mode"] == "Mode::Bicycle"], show_vertices=False, color="Purple", layer_name="bicycle")
    show_trajectories(m, df[df["mode"] == "Mode::Boat"], show_vertices=False, color="Green", layer_name="boat")
    show_trajectories(m, df[df["mode"] == "Mode::Walk"], show_vertices=False, color="HotPink", layer_name="walk")


def show_points(m, df, column_name=False, color="Black", layer_name="points"):
    """Draw points on leaflet map"""
    points = list()

    for ix, row in df.iterrows():
        message = HTML()
        message.value = "not labeled"
        if column_name:
            message.value = row[column_name]

        circle = ipy.CircleMarker()
        circle.location = (row.geometry.y, row.geometry.x)
        circle.radius = 5
        circle.fill_opacity = 0.6
        circle.fill_color = "Gray"
        circle.stroke = True
        circle.color = color
        circle.weight = 1
        circle.opacity = 0.3
        circle.popup = message
        points.append(circle)

    layer_points = ipy.LayerGroup(layers=points)
    layer_points.name = layer_name

    m.add_layer(layer_points)
