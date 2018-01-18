import logging
import pyproj
import http.client as http_client
from PIL import Image

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

DIRECTORY = '/Users/kellens/Temp/Tulips/'
WMS_INSTANCE = '71513b0b-264d-494a-b8c4-c3c36433db28'
layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}

box_height = abs(526803.9989414361 - 531695.9687516851)
box_width = abs(6966795.615281175 - 6971687.585091426)
# Initialize me to bottom left
starting_bottom_left = to_epsg3857((52.89646, 4.72131))
starting_top_right = (starting_bottom_left[0] + box_height, starting_bottom_left[1] + box_width)
bbox = [starting_bottom_left, starting_top_right]


def save_image(npdata, outfilename):
    img = Image.fromarray(npdata)
    img.save(outfilename)


for i in range(0, 10):
    for v in range(0, 10):
        tulip_fields = TulipFieldRequest(bbox=bbox, width=512, height=512, crs=3857, layer=layers['tulip_field_2016'])
        tulip_data = tulip_fields.get_data()
        save_image(tulip_data[0], '{}{}_{}_tulips_2016.bmp'.format(DIRECTORY, i, v))

        s2_request = S2Request(WMS_INSTANCE, layers='TRUE_COLOR', time=('2016-05-01'), bbox=bbox,
                               width=512, height=512, crs=3857)
        sat_data = s2_request.get_data()
        save_image(sat_data[0], '{}{}_{}_sat_2016_05_01.bmp'.format(DIRECTORY, i, v))
        bbox = [(bbox[0][0], bbox[0][1] + box_width), (bbox[1][0], bbox[1][1] + box_width)]
    bbox = [(bbox[0][0] + box_height, bbox[0][1]), (bbox[1][0] + box_height, bbox[1][1])]
