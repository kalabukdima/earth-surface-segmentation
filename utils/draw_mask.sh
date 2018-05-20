#!/bin/bash

if [[ $# < 2 ]]; then
    echo 'Usage: sh draw_mask.sh filename.shp.zip image.tif [gray]'
    exit
fi

if [[ $3 == 'gray' ]]; then
    forest=(0 0 0)
    meadow=(1 1 1)
    water=(2 2 2)
    settlement=(3 3 3)
    none=(255 255 255)
else
    forest=(0 255 0)
    meadow=(255 255 0)
    settlement=(255 0 0)
    water=(0 0 255)
    none=(255 255 255)
fi

cp $2 out.tif

# Erase image (slow)
gdal_rasterize -i -b 1 -b 2 -b 3 -burn ${none[0]} -burn ${none[1]} -burn ${none[2]} -where\
    "fclass='nonexistent_class'" /vsizip/$1/gis.osm_places_a_free_1.shp out.tif

# Burn all the layers
gdal_rasterize -b 1 -b 2 -b 3 -burn ${meadow[0]} -burn ${meadow[1]} -burn ${meadow[2]} -where\
    "fclass='wetland'" /vsizip/$1/gis.osm_water_a_free_1.shp out.tif

gdal_rasterize -b 1 -b 2 -b 3 -burn ${settlement[0]} -burn ${settlement[1]} -burn ${settlement[2]} -where\
    "fclass='city' OR fclass='town' OR fclass='village' OR fclass='suburb' OR fclass='hamlet'" /vsizip/$1/gis.osm_places_a_free_1.shp out.tif

gdal_rasterize -b 1 -b 2 -b 3 -burn ${settlement[0]} -burn ${settlement[1]} -burn ${settlement[2]} -where\
    "fclass='residential' OR fclass='industrial' OR fclass='allotments'" /vsizip/$1/gis.osm_landuse_a_free_1.shp out.tif

gdal_rasterize -b 1 -b 2 -b 3 -burn ${forest[0]} -burn ${forest[1]} -burn ${forest[2]} -where\
    "fclass='forest'" /vsizip/$1/gis.osm_landuse_a_free_1.shp out.tif

gdal_rasterize -b 1 -b 2 -b 3 -burn ${meadow[0]} -burn ${meadow[1]} -burn ${meadow[2]} -where\
    "fclass='farm' OR fclass='meadow' OR fclass='orchard' OR fclass='heath' OR fclass='scrub' OR fclass='grass'" /vsizip/$1/gis.osm_landuse_a_free_1.shp out.tif

gdal_rasterize -b 1 -b 2 -b 3 -burn ${forest[0]} -burn ${forest[1]} -burn ${forest[2]} -where\
    "fclass='park'" /vsizip/$1/gis.osm_landuse_a_free_1.shp out.tif

gdal_rasterize -b 1 -b 2 -b 3 -burn ${water[0]} -burn ${water[1]} -burn ${water[2]} -where\
    "fclass='water' OR fclass='river' OR fclass='reservoir'" /vsizip/$1/gis.osm_water_a_free_1.shp out.tif

