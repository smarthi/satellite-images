import logging
import pyproj
import os
import http.client as http_client
from PIL import Image
from dateutil import parser
from urllib import parse #from urlparse import urlparse

from DataRequest import TulipFieldRequest, S2Request

http_client.HTTPConnection.debuglevel = 1
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True


def to_epsg3857(latlong_wgs84):
    epsg3857 = pyproj.Proj(init='epsg:3857')
    wgs84 = pyproj.Proj(init='EPSG:4326')
    return pyproj.transform(wgs84,epsg3857,latlong_wgs84[1],latlong_wgs84[0])


def to_wgs84(latlong_epsg3857):
    wgs84 = pyproj.Proj(init='EPSG:4326')
    epsg3857 = pyproj.Proj(init='epsg:3857')
    res = pyproj.transform(epsg3857,wgs84,latlong_epsg3857[0],latlong_epsg3857[1])
    return [res[1], res[0]]

DIRECTORY = 'images/'
WMS_INSTANCE = '71513b0b-264d-494a-b8c4-c3c36433db28'
layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}


box_height = 1500
box_width = 1500
sat_map_layer = 'TRUE_COLOR'
# Initialize me to bottom left and go to top right
bottom_left = to_epsg3857((52.65097891432322, 4.717941284179688))
top_right = to_epsg3857((52.86374108718206, 5.069503784179688))
starting_top_right = (bottom_left[0] + box_height, bottom_left[1] + box_width)
bbox = [bottom_left, starting_top_right]

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)


def save_image(npdata, outfilename):
    img = Image.fromarray(npdata)
    img.save(outfilename)


for i in range(0, int(abs((bottom_left[0]-top_right[0])/box_height))):
    temp_width = [bbox[0][1], bbox[1][1]]
    for v in range(0, int(abs((bottom_left[1]-top_right[1])/box_width))):
        logging.debug('Loading {},{}'.format(to_wgs84(bbox[0])[0], to_wgs84(bbox[0])[1]))
        tulip_fields = TulipFieldRequest(bbox=bbox, width=512, height=512, crs=3857, layer=layers['tulip_field_2016'])
        tulip_data = tulip_fields.get_data()
        save_image(tulip_data[0], '{}{}_{}_{}_{}_tulips_2016.bmp'.format(DIRECTORY, int(bbox[0][0]), int(bbox[0][1]), int(bbox[1][0]), int(bbox[1][1])))

        s2_request = S2Request(WMS_INSTANCE, layers=sat_map_layer, time=('2016-03-01', '2017-10-20'), bbox=bbox,
                               width=512, height=512, crs=3857, maxcc=0.5)
        sat_data = s2_request.get_data()
        dates = [parser.parse(parse.parse_qs(parse.urlparse(item.url).query)['time'][0].split('/')[0])
                 for item in s2_request.download_list]
        for i, date in enumerate(dates):
            if sat_data:
                save_image(sat_data[i], '{}{}_{}_{}_{}_sat_{}_{}.bmp'.format(DIRECTORY, int(bbox[0][0]), int(bbox[0][1]), int(bbox[1][0]), int(bbox[1][1]), date.isoformat(), sat_map_layer))
        bbox = [(bbox[0][0], bbox[0][1] + box_width), (bbox[1][0], bbox[1][1] + box_width)]
    bbox = [(bbox[0][0] + box_height, temp_width[0]), (bbox[1][0] + box_height, temp_width[1])]