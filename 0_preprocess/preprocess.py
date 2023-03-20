import pandas as pd

import os

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import trackintel as ti


def filter_user(tl):
    # TODO: specify staypoint file to read
    sp = ti.io.file.read_staypoints_csv(
        os.path.join("data", "staypoints.csv"), index_col="index", geom_col="geom", crs=4326
    )
    # sp from original data, example data format:
    # index, id, user_id, started_at, finished_at
    # 0, 1, 2, 2016-03-20 09:16:29+00:00, 2016-03-20 09:46:29+00:00
    # 1, 3, 3, 2017-03-19 09:52:15+00:00, 2017-03-19 13:21:58+00:00

    # calculate temporal tracking quality per user
    tl_sp = pd.concat([tl, sp])
    user_tracking_quality = ti.analysis.tracking_quality.temporal_tracking_quality(tl_sp, granularity="all")
    user_tracking_quality.rename(columns={"quality": "user_t_quality"}, inplace=True)
    user_tracking_quality

    tl = tl.merge(user_tracking_quality, on="user_id", how="left")

    tl = tl[tl["user_t_quality"] >= 0.5]

    return tl, len(tl)


def filter_area(tl):
    # TODO: specify area file to read
    CH = gpd.read_file(os.path.join("data", "swiss", "swiss.shp"))
    CH_WGS84 = CH.to_crs(4326)

    # simplify boundaries with Douglas Peuker for faster filtering
    CH_generalized = CH_WGS84.copy()
    CH_generalized["geometry"] = CH_generalized["geometry"].simplify(0.005)
    # count_before = len(CH_WGS84.iloc[0]["geometry"].exterior.coords)  # count number of coordinate points
    # count_after = len(CH_generalized.iloc[0]["geometry"].exterior.coords)  # count number of coordinate points
    # print(round(((1 - count_after / count_before) * 100), 1), "percent compressed")

    # select triplegs within Switzerland
    tl = ti.preprocessing.filter.spatial_filter(tl, CH_generalized, method="within")
    return tl, len(tl)


def filter_mode(tl):
    # remove triplegs with trip modes: plane, coach and ski
    exclude = (tl["mode"] == "Mode::Airplane") | (tl["mode"] == "Mode::Coach") | (tl["mode"] == "Mode::Ski")
    tl = tl.drop(tl[exclude].index)

    return tl, len(tl)


def filter_legs(tl):
    # remove triplegs with less than 50 metres or with less than 60 seconds
    exclude = (tl["length"] < 50) | (tl["dur_s"] < 60)
    tl = tl.drop(tl[exclude].index)
    return tl, len(tl)


def filter_quality(tl):
    # calculate average recording interval
    tl["recording_interval"] = tl["dur_s"] / [(len(row.coords) - 1) for row in tl["geom"]]
    # select triplegs with more than 4 track points and with average recording interval less than 60 seconds
    exclude = [(len(s.coords)) < 4 for s in tl["geom"]] | (tl["recording_interval"] > 60)
    tl = tl.drop(tl[exclude].index)
    return tl, len(tl)


def filter_speed(tl):
    # calculate speed
    tl["speed_av"] = tl["length"] / tl["dur_s"] * 3.6
    exclude = (
        ((tl["speed_av"] > 20) & (tl["mode"] == "Mode::Walk"))
        | ((tl["speed_av"] > 60) & (tl["mode"] == "Mode::Bicycle"))
        | ((tl["speed_av"] > 60) & (tl["mode"] == "Mode::Ebicycle"))
        | ((tl["speed_av"] > 250) & (tl["mode"] == "Mode::Train"))
        | ((tl["speed_av"] > 80) & (tl["mode"] == "Mode::Tram"))
        | ((tl["speed_av"] > 150) & (tl["mode"] == "Mode::Bus"))
        | ((tl["speed_av"] > 50) & (tl["mode"] == "Mode::Boat"))
        | ((tl["speed_av"] > 150) & (tl["mode"] == "Mode::Car"))
        | ((tl["speed_av"] > 150) & (tl["mode"] == "Mode::Ecar"))
    )
    tl = tl.drop(tl[exclude].index)
    return tl, len(tl)


if __name__ == "__main__":
    # load triplegs and staypoints from CSV
    tl = ti.io.file.read_triplegs_csv(
        os.path.join("data", "triplegs.csv"), index_col="index", geom_col="geom", crs=4326
    )
    tl["dur_s"] = (tl["finished_at"] - tl["started_at"]).dt.total_seconds()
    n_tl_raw = len(tl)
    # tl from original data, example data format:
    # index, id, user_id, started_at, finished_at, mode, geom, length
    # 0, 29, 16, 2016-04-14 18:17:23.000+00:00, 2016-04-14 18:20:57.000+00:00, Mode::Walk, "LINESTRING (8.0125 47.5064, 8.0129 47.50666)", 235.0
    # 1, 32, 17, 2017-04-30 11:05:40.000+00:00, 2017-04-30 11:19:45.000+00:00, Mode::Walk, "LINESTRING (6.6961 46.8441, 6.6962 46.8446)", 1035.4

    # for filtering user, staypoints need to be pre-specified.
    print("Filtering user")
    tl, n_tl_user_selection = filter_user(tl)

    # for area filter, area shp need to be specified
    print("Filtering area")
    tl, n_tl_within_CH = filter_area(tl)

    # check the mode list and specify which mode to exclude
    print("Filtering invalid modes")
    tl, n_tl_mode_selection = filter_mode(tl)

    print("Filtering short triplegs")
    tl, n_tl_length_duration_selection = filter_legs(tl)

    print("Filtering low-quality triplegs")
    tl, n_tl_tracking_quality_selection = filter_quality(tl)

    print("Filtering high-speed triplegs")
    tl, n_tl_speed_selection = filter_speed(tl)

    # show number of trip legs after each step
    print(
        "#triplegs original data:                      ",
        n_tl_raw,
        "\n" "#triplegs after step 1:                       ",
        n_tl_user_selection,
        "\n" "#triplegs after step 2:                       ",
        n_tl_within_CH,
        "\n" "#triplegs after step 3:                       ",
        n_tl_mode_selection,
        "\n" "#triplegs after step 4:                       ",
        n_tl_length_duration_selection,
        "\n" "#triplegs after step 5:                       ",
        n_tl_tracking_quality_selection,
        "\n" "#triplegs after step 6 (pre-processed data):  ",
        n_tl_speed_selection,
    )
    # triplegs original data:                       457754
    # triplegs after step 1:                        447695
    # triplegs after step 2:                        397593
    # triplegs after step 3:                        395932
    # triplegs after step 4:                        376866
    # triplegs after step 5:                        371218
    # triplegs after step 6 (pre-processed data):   365307

    print(tl["mode"].value_counts())

    # Mode::Walk        155177
    # Mode::Ecar         78758
    # Mode::Car          51920
    # Mode::Train        51470
    # Mode::Bicycle      11575
    # Mode::Bus           9436
    # Mode::Tram          6231
    # Mode::Ebicycle       373
    # Mode::Boat           367
    # Name: mode, dtype: int64

    # save pre-processed trip legs
    tl.to_csv(os.path.join("data", "triplegs_preprocessed.csv"), index_label="index")
