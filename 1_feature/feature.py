import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import os

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import trackintel as ti
from shapely.geometry import Point
from scipy.spatial import KDTree
import statistics


def _calculate_length(points):
    """Calculate length of intervals"""
    lengths = np.linalg.norm(np.array(points.coords)[1:] - np.array(points.coords)[:-1], axis=1)
    return lengths


def _calculate_bearing(points):
    """Calculate bearing (heading direction) of intervals"""
    interval = np.array(points.coords)[1:] - np.array(points.coords)[:-1]
    return np.arctan2(interval[:, 0], interval[:, 1])


def _calculate_bearing_rate(bearings):
    """Calculate bearing rate (change in heading direction) between intervals"""
    diff = bearings[1:] - bearings[:-1]
    return np.minimum(diff % (2 * np.pi), ((2 * np.pi) - diff) % (2 * np.pi))


def _calculate_speed(row):
    """Calculate speed of intervals"""
    return np.array([length / row["recording_interval"] for length in row["lengths_of_intervals"]])


def _calculate_acceleration(row):
    """Calculate acceleration between intervals"""
    return abs((row["speeds_of_intervals"][1:] - row["speeds_of_intervals"][:-1]) / row["recording_interval"])


def _create_index(df, geom_type):
    """Create spatial index. For point features return sindex and kdtree, otherwise only sindex."""
    # transform from WGS84 to LV95
    df_LV95 = df.to_crs(2056)
    # create sindex
    df_LV95_sindex = df_LV95.sindex
    if geom_type != "point":
        return df_LV95, df_LV95_sindex
    else:
        # create kdtree
        df_LV95_kdtree = KDTree(np.squeeze(np.array([point.coords for point in df_LV95.geometry])))
        return df_LV95, df_LV95_sindex, df_LV95_kdtree


def _compute_startEnd_feature(tl, feature_kdtree):
    """Calculate min/max Euclidean distance from start/end point to closest geospatial object"""
    tl["feature_start"] = feature_kdtree.query(np.array(tl["start_point"].tolist()), k=[1])[0]
    tl["feature_end"] = feature_kdtree.query(np.array(tl["end_point"].tolist()), k=[1])[0]
    tl["feature_min"] = tl[["feature_start", "feature_end"]].values.min(1)
    tl["feature_max"] = tl[["feature_start", "feature_end"]].values.max(1)
    return tl[["feature_min", "feature_max"]]


def _compute_trackPoints_point_feature(tl_geom, feature_kdtree, n=5):
    """Calculate average Euclidean distance of track points to closest geospatial point object"""

    # extract every n-th track point of trajectory (simplification with regular time intervals)
    points_list = [np.array([x, y]) for x, y in tl_geom.coords[::n]]

    # calculate distance to closest object for every track point
    dists = feature_kdtree.query(np.array(points_list), k=[1])[0]

    # calculate average distance
    return dists.mean()


def _compute_trackPoints_line_feature(tl_geom, feature_gdf, feature_sindex, n=5):
    """Calculate average Euclidean distance of track points to closest point on network"""

    # extract every n-th track point of trajectory (simplification with regular time intervals)
    points_list = [Point(x, y) for x, y in tl_geom.coords[::n]]
    points_gs = gpd.GeoSeries(points_list)

    # find closest object for every track point
    indices = feature_sindex.nearest(points_gs, return_all=False)

    # calculate distance to closest object for every track point
    dists = [feature_gdf.geometry.iloc[y].distance(points_gs[x]) for x, y in zip(indices[0], indices[1])]

    # calculate average distance
    return statistics.mean(dists)


def _compute_trackPoints_polygon_feature(tl_geom, feature_sindex, n=5):
    """Calculate proportion of overlap of track points to land cover objects"""

    # extract every n-th track point of trajectory (simplification with regular time intervals)
    points_list = [Point(x, y) for x, y in tl_geom.coords[::n]]
    points_gs = gpd.GeoSeries(points_list)

    # check if track point lies within object
    indices_within = feature_sindex.query_bulk(points_gs, predicate="within")

    # calculate proportion of track points lieing within object
    n_total = len(points_gs)
    n_within = len(np.unique(indices_within[0]))
    return n_within / n_total


def _apply_parallel(df, func, n=-1, **kwargs):
    """parallel apply for spending up."""
    if n is None:
        n = -1

    cpunum = 20
    dflength = len(df)
    if dflength < cpunum:
        spnum = dflength
    if n < 0:
        spnum = cpunum + n + 1
    else:
        spnum = n or 1

    sp = list(range(dflength)[:: int(dflength / spnum + 0.5)])
    sp.append(dflength)
    slice_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))

    results = Parallel(n_jobs=n, verbose=0)(delayed(func)(df.iloc[slc], **kwargs) for slc in slice_gen)
    return pd.concat(results)


def _apply_extract_line(tl, **kwargs):
    tqdm.pandas(desc="Line")
    return tl.geom_LV95.progress_apply(lambda x: _compute_trackPoints_line_feature(x, n=1, **kwargs))


def _apply_extract_polygon(tl, **kwargs):
    tqdm.pandas(desc="Polygon")
    return tl.geom_LV95.progress_apply(lambda x: _compute_trackPoints_polygon_feature(x, n=1, **kwargs))


def motion_features(tl):
    # calculate properties of every interval
    tqdm.pandas(desc="Bar")
    # calculate lengths [m]
    tl["lengths_of_intervals"] = tl["intervals"].progress_apply(_calculate_length)
    # calculate speeds [m/s]
    tl["speeds_of_intervals"] = tl.progress_apply(_calculate_speed, axis=1)
    # calculate accelerations [m/(s^2)]
    tl["acc_of_intervals"] = tl.progress_apply(_calculate_acceleration, axis=1)
    # calculate bearings [rad] within (-pi,pi]
    tl["bearings_of_intervals"] = tl["intervals"].progress_apply(_calculate_bearing)
    # calculate absolute value of minimal angle between bearings [rad] within [0,pi]
    tl["br_of_intervals"] = tl["bearings_of_intervals"].progress_apply(_calculate_bearing_rate)

    # calculate 85th percentile of speed based on speeds of elementar intervals
    tl["speed_85th"] = [np.percentile(speeds, 85) * 3.6 for speeds in tl["speeds_of_intervals"]]
    # calculate average acceleraton based on acceleration of elementar intervals
    tl["acc_av"] = [np.average(acc) for acc in tl["acc_of_intervals"]]
    tl["acc_85th"] = [np.percentile(acc, 85) for acc in tl["acc_of_intervals"]]

    # calculate average bearing rate based on bearing rates of elementar intervals
    tl["br_av"] = [np.average(br) for br in tl["br_of_intervals"]]
    tl["br_85th"] = [np.percentile(br, 85) for br in tl["br_of_intervals"]]

    return tl


def context_features(tl):
    # TODO: context data files
    # load OSM point layers
    OSM_transport = gpd.read_file("data/osm_switzerland_17-01-01/gis_osm_transport_free_1.shp")
    OSM_traffic = gpd.read_file("data/osm_switzerland_17-01-01/gis_osm_traffic_free_1.shp")
    OSM_poi = gpd.read_file("data/osm_switzerland_17-01-01/gis_osm_pois_free_1.shp")
    OSM_pofw = gpd.read_file("data/osm_switzerland_17-01-01/gis_osm_pofw_free_1.shp")

    # load OSM line layers
    OSM_roads = gpd.read_file("data/osm_switzerland_17-01-01/gis_osm_roads_free_1.shp")
    OSM_railways = gpd.read_file("data/osm_switzerland_17-01-01/gis_osm_railways_free_1.shp")

    # load OSM polygon layers
    OSM_water = gpd.read_file("data/osm_switzerland_latest/gis_osm_water_a_free_1.shp")
    OSM_landuse = gpd.read_file("data/osm_switzerland_latest/gis_osm_landuse_a_free_1.shp")

    # load swissTLMregio rivers
    v25_rivers = gpd.read_file("data/oberflachengewasser_vector25/Typisierung_LV95/FGT.shp")

    print("Context data loading complete")

    # extract railway stations
    rw_stations = OSM_transport[
        ((OSM_transport["fclass"] == "railway_station") | (OSM_transport["fclass"] == "railway_halt"))
    ].copy()
    _, _, rw_stations_LV95_kdtree = _create_index(rw_stations, "point")

    # extract tram stops
    tram_stops = OSM_transport[(OSM_transport["fclass"] == "tram_stop")].copy()
    _, _, tram_stops_LV95_kdtree = _create_index(tram_stops, "point")

    # extract bus stops
    bus_stops = OSM_transport[
        ((OSM_transport["fclass"] == "bus_stop") | (OSM_transport["fclass"] == "bus_station"))
    ].copy()
    _, _, bus_stops_LV95_kdtree = _create_index(bus_stops, "point")

    # extract car parkings
    car_parking_types = ["parking", "parking_site", "parking_multistorey", "parking_underground"]
    car_parkings = OSM_traffic.loc[OSM_traffic["fclass"].isin(car_parking_types)].copy()
    _, _, car_parkings_LV95_kdtree = _create_index(car_parkings, "point")

    # extract bycicle parkings
    bicycle_parkings = OSM_traffic[OSM_traffic["fclass"] == "parking_bicycle"].copy()
    _, _, bicycle_parkings_LV95_kdtree = _create_index(bicycle_parkings, "point")

    # extract landing stages
    ferry_terminals = OSM_transport[OSM_transport["fclass"] == "ferry_terminal"]
    harbours = OSM_traffic[OSM_traffic["fclass"] == "marina"]
    landing_stages = pd.concat([ferry_terminals, harbours])
    landing_stages.reset_index(inplace=True)
    _, _, landing_stages_LV95_kdtree = _create_index(landing_stages, "point")

    # extract points of interest
    non_poi_types = [
        "fire_station",
        "prison",
        "hunting_stand",
        "camera_surveillance",
        "emergency_phone",
        "emergency_access",
        "tower_comms",
        "water_tower",
        "tower_observation",
        "waste_basket",
        "wastewater_plant",
        "water_works",
    ]
    poi = OSM_poi.loc[~OSM_poi["fclass"].isin(non_poi_types)].copy()
    poi = pd.concat([poi, OSM_pofw])
    poi.reset_index(inplace=True)
    _, _, poi_LV95_kdtree = _create_index(poi, "point")

    # extract railway network
    rw_types = ["rail", "light_rail", "narrow_gauge", "rack"]
    rw_network = OSM_railways.loc[OSM_railways["fclass"].isin(rw_types)].copy()
    rw_network_LV95, rw_network_LV95_sindex = _create_index(rw_network, "line")

    # extract tram network
    tram_types = ["tram", "subway"]
    tram_network = OSM_railways.loc[OSM_railways["fclass"].isin(tram_types)].copy()
    tram_network_LV95, tram_network_LV95_sindex = _create_index(tram_network, "line")

    # extract road network
    road_types = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    ]
    road_network = OSM_roads.loc[OSM_roads["fclass"].isin(road_types)].copy()
    road_network_LV95, road_network_LV95_sindex = _create_index(road_network, "line")

    # extract pedestrian and bicycle network
    pb_types = [
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "pedestrian",
        "service",
        "track",
        "track_grade1",
        "track_grade2",
        "track_grade3",
        "track_grade4",
        "track_grade5",
        "cycleway",
        "footway",
        "path",
        "steps",
    ]
    pb_network = OSM_roads.loc[OSM_roads["fclass"].isin(pb_types)].copy()
    pb_network_LV95, pb_network_LV95_sindex = _create_index(pb_network, "line")

    # extract large water bodies
    # select lakes and large rivers
    lakes = OSM_water[((OSM_water["fclass"] == "water") | (OSM_water["fclass"] == "reservoir"))].copy()
    rivers = v25_rivers[v25_rivers["GROSSERFLU"] != "NA"].copy()
    # filter lakes based on area and simplify borders with Douglas Peuker
    lakes.to_crs(2056, inplace=True)
    lakes = lakes[lakes["geometry"].area >= 100000]
    lakes["geometry"] = lakes["geometry"].simplify(0.5)
    lakes.to_crs(4326, inplace=True)
    # buffer rivers (from line to polygon) and simplify borders with Douglas Peuker
    rivers.rename(columns={"OBJECTID_G": "id", "GROSSERFLU": "name"}, inplace=True)
    rivers = rivers[["id", "name", "geometry"]].copy()
    rivers.reset_index(inplace=True)
    rivers["geometry"] = rivers["geometry"].buffer(40)
    rivers["geometry"] = rivers["geometry"].simplify(0.5)
    rivers.to_crs(4326, inplace=True)

    # merge lakes and rivers
    water_bodies = pd.concat([lakes, rivers])
    water_bodies.reset_index(inplace=True)
    _, water_bodies_LV95_sindex = _create_index(water_bodies, "polygon")

    # extract public green spaces
    green_space_types = ["park", "cemetery", "recreation_ground"]
    green_spaces = OSM_landuse.loc[OSM_landuse["fclass"].isin(green_space_types)].copy()
    _, green_spaces_LV95_sindex = _create_index(green_spaces, "polygon")

    # extract residental areas
    residental_areas = OSM_landuse[(OSM_landuse["fclass"] == "residential")].copy()
    _, residental_areas_LV95_sindex = _create_index(residental_areas, "polygon")

    # extract forest areas
    forest_areas = OSM_landuse[(OSM_landuse["fclass"] == "forest")].copy()
    _, forest_areas_LV95_sindex = _create_index(forest_areas, "polygon")

    print("Context data processing complete")

    # add start and end coordinates to triplegs
    tl["start_geom"] = [Point(segment.coords[0]) for segment in tl["geom_LV95"]]
    tl["end_geom"] = [Point(segment.coords[-1]) for segment in tl["geom_LV95"]]
    tl["start_point"] = [np.array(segment.coords)[0] for segment in tl["geom_LV95"]]
    tl["end_point"] = [np.array(segment.coords)[-1] for segment in tl["geom_LV95"]]

    # endpoint features
    print("Extracting endpoint features")
    tl[["rwStation_se_dist_min", "rwStation_se_dist_max"]] = _compute_startEnd_feature(tl, rw_stations_LV95_kdtree)
    tl[["tramStop_se_dist_min", "tramStop_se_dist_max"]] = _compute_startEnd_feature(tl, tram_stops_LV95_kdtree)
    tl[["busStop_se_dist_min", "busStop_se_dist_max"]] = _compute_startEnd_feature(tl, bus_stops_LV95_kdtree)
    tl[["carParking_se_dist_min", "carParking_se_dist_max"]] = _compute_startEnd_feature(tl, car_parkings_LV95_kdtree)
    tl[["bicycleParking_se_dist_min", "bicycleParking_se_dist_max"]] = _compute_startEnd_feature(
        tl, bicycle_parkings_LV95_kdtree
    )
    tl[["landingStage_se_dist_min", "landingStage_se_dist_max"]] = _compute_startEnd_feature(
        tl, landing_stages_LV95_kdtree
    )

    # point object features
    print("Extracting point object features")
    tl["rwStations_dist_av"] = tl.geom_LV95.apply(
        lambda x: _compute_trackPoints_point_feature(x, rw_stations_LV95_kdtree, n=1)
    )
    tl["tramStops_dist_av"] = tl.geom_LV95.apply(
        lambda x: _compute_trackPoints_point_feature(x, tram_stops_LV95_kdtree, n=1)
    )
    tl["busStops_dist_av"] = tl.geom_LV95.apply(
        lambda x: _compute_trackPoints_point_feature(x, bus_stops_LV95_kdtree, n=1)
    )
    tl["poi_dist_av"] = tl.geom_LV95.apply(lambda x: _compute_trackPoints_point_feature(x, poi_LV95_kdtree, n=1))

    # network features
    print("Extracting network features")
    tl["rwNetwork_dist_av"] = _apply_parallel(
        tl, _apply_extract_line, n=-3, feature_gdf=rw_network_LV95, feature_sindex=rw_network_LV95_sindex
    )
    print("Finish rail network feature")
    tl["tramNetwork_dist_av"] = _apply_parallel(
        tl, _apply_extract_line, n=-3, feature_gdf=tram_network_LV95, feature_sindex=tram_network_LV95_sindex
    )
    print("Finish tram network feature")
    tl["roadNetwork_dist_av"] = _apply_parallel(
        tl, _apply_extract_line, n=-3, feature_gdf=road_network_LV95, feature_sindex=road_network_LV95_sindex
    )
    print("Finish road network feature")
    tl["pbNetwork_dist_av"] = _apply_parallel(
        tl, _apply_extract_line, n=-3, feature_gdf=pb_network_LV95, feature_sindex=pb_network_LV95_sindex
    )
    print("Finish ped network feature")

    # LULC featues
    print("Extracting LULC features")
    tl["water_ovlp_prop"] = _apply_parallel(tl, _apply_extract_polygon, n=-3, feature_sindex=water_bodies_LV95_sindex)
    print("Finish water body feature")
    tl["greenSpaces_ovlp_prop"] = _apply_parallel(
        tl, _apply_extract_polygon, n=-3, feature_sindex=green_spaces_LV95_sindex
    )
    print("Finish green space feature")
    tl["residental_ovlp_prop"] = _apply_parallel(
        tl, _apply_extract_polygon, n=-3, feature_sindex=residental_areas_LV95_sindex
    )
    print("Finish residential space feature")
    tl["forest_ovlp_prop"] = _apply_parallel(tl, _apply_extract_polygon, n=-3, feature_sindex=forest_areas_LV95_sindex)
    print("Finish forest overlap feature")

    return tl


if __name__ == "__main__":
    # TODO: specify tripleg file from preprocess.py script
    tl = ti.io.file.read_triplegs_csv(
        os.path.join("data", "triplegs_preprocessed.csv"), index_col="index", geom_col="geom", crs=4326
    )
    # tl from preprocessing script, example data format:
    # index, id, user_id, started_at, finished_at, mode, geom, length, dur_s, user_t_quality, recording_interval, speed_av
    # 0, 29, 16, 2016-04-14 18:17:23.000+00:00, 2016-04-14 18:20:57.000+00:00, Mode::Walk, "LINESTRING (8.0125 47.5064, 8.0129 47.50666)", 235.018, 213.999, 0.9182, 7.1333, 3.953
    # 1, 32, 17, 2017-04-30 11:05:40.000+00:00, 2017-04-30 11:19:45.000+00:00, Mode::Walk, "LINESTRING (6.6961 46.8441, 6.6962 46.8446)", 1035.4314, 845.063, 0.859, 16.569, 4.410

    # transform coordinates from WGS84 to LV95
    tl["geom_LV95"] = tl["geom"]
    tl.set_geometry("geom_LV95", inplace=True)
    tl.to_crs(2056, inplace=True)

    tl.rename(columns={"dur_s": "dur"}, inplace=True)

    # add list of intervals to each tripleg
    tl["intervals"] = tl["geom_LV95"].apply(np.array)

    # motion feature extraction
    tl = motion_features(tl)

    # for context features, osm feature layers need to be stored in the data folder
    tl = context_features(tl)

    # final cleaning and safe
    tl = tl[
        [
            "user_id",
            "mode",
            "started_at",
            "finished_at",
            "user_t_quality",
            "recording_interval",
            "length",
            "dur",
            "speed_av",
            "speed_85th",
            "acc_av",
            "acc_85th",
            "br_av",
            "br_85th",
            "rwStation_se_dist_min",
            "rwStation_se_dist_max",
            "tramStop_se_dist_min",
            "tramStop_se_dist_max",
            "busStop_se_dist_min",
            "busStop_se_dist_max",
            "carParking_se_dist_min",
            "carParking_se_dist_max",
            "bicycleParking_se_dist_min",
            "bicycleParking_se_dist_max",
            "landingStage_se_dist_min",
            "landingStage_se_dist_max",
            "rwStations_dist_av",
            "tramStops_dist_av",
            "busStops_dist_av",
            "poi_dist_av",
            "rwNetwork_dist_av",
            "tramNetwork_dist_av",
            "roadNetwork_dist_av",
            "pbNetwork_dist_av",
            "water_ovlp_prop",
            "greenSpaces_ovlp_prop",
            "residental_ovlp_prop",
            "forest_ovlp_prop",
        ]
    ]
    tl.to_csv(os.path.join("data", "triplegs_features_temp.csv"), index_label="index")
