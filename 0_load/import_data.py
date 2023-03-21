import psycopg2
import json

import os

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import trackintel as ti


if __name__ == "__main__":
    # database connections
    # TODO: define your own database connection file and parameters
    DBLOGIN_FILE = os.path.join(".", "credentials.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    conn = psycopg2.connect(**CONFIG)

    # read data
    # load triplegs from database
    tpls_str = (
        "SELECT id, user_id, started_at, finished_at,"
        "mode_detected, mode_validated, validated, geometry,"
        "ST_Length(geometry::geography) As length FROM v1.triplegs WHERE ST_NumPoints(geometry) > 1"
    )

    tpls_raw = gpd.GeoDataFrame.from_postgis(
        tpls_str, conn, geom_col="geometry", crs=4326, parse_dates=["started_at", "finished_at"]
    )

    tpls_raw = tpls_raw.loc[tpls_raw.geometry.is_valid]

    # load staypoints from database
    sp_raw = ti.io.postgis.read_staypoints_postgis(
        "SELECT * FROM v1.staypoints", conn, geom_col="geometry_raw", crs=4326
    )

    sp_raw.rename(columns={"geometry_raw": "geom"}, inplace=True)

    # save
    if not os.path.exists("data"):
        os.makedirs("data")
    # write triplegs to CSV
    tpls_raw.to_csv(os.path.join("data", "triplegs.csv"), index_label="index")
    # write staypoints to CSV
    sp_raw.to_csv(os.path.join("data", "staypoints.csv"), index_label="index")
