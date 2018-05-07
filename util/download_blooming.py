'''
Script to download the blooming tulips dataset from sentinelhub
'''
import sys
sys.path.append('../DataRequest/')
from gather_data import PolygonSlidingWindow, GeoJsonSaver, BatchDownloader
from DataRequest import TulipFieldRequest, S2Request

WMS_INSTANCE = '71513b0b-264d-494a-b8c4-c3c36433db28'
geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}

poly_sw_cb = PolygonSlidingWindow(box_width=2560, box_height=2560, stride_x=2304, stride_y=2304)
poly_sw_cb.set_mode('train')
poly_sw_cb.load_polygons_from_folder('../examples/tulip-polygons/')

# Check how many patches are contained in the defined polygons
patches = list(poly_sw_cb.patches.values())[0]
patches = [item for sublist in patches for item in sublist]
print("Patches found: {}".format(len(patches)))

root_dir = '../data/tulips/blooming/'

# Params
width = 256
height = 256
maxcc = 20

# 2016
# Download labels
# poly_sw_cb.set_mode('train')
# downloader_train_labels = BatchDownloader(root_dir + '2016/', poly_sw_cb, TulipFieldRequest, (),
# 							 {'width':width, 'height':height, 'crs':3857, 'layer':geopedia_layers['tulip_field_2016']})
# downloader_train_labels.download_data()

# Download images
poly_sw_cb.set_mode('train')
downloader_train_images = BatchDownloader(root_dir + '2016/', poly_sw_cb,
                             S2Request, (WMS_INSTANCE,),
                             {'width':width, 'height':height, 'crs':3857, 'time':('2016-04-10','2016-05-10'), 'layers':'TRUE_COLOR', 'maxcc':maxcc})
downloader_train_images.download_data()

# 2017
# # Download labels
# poly_sw_cb.set_mode('train')
# downloader_train_labels = BatchDownloader(root_dir + '2017/', poly_sw_cb, TulipFieldRequest, (),
# 							 {'width':width, 'height':height, 'crs':3857, 'layer':geopedia_layers['tulip_field_2017']})
# downloader_train_labels.download_data()

# # Download images
# poly_sw_cb.set_mode('train')
# downloader_train_images = BatchDownloader(root_dir + '2017/', poly_sw_cb,
#                              S2Request, (WMS_INSTANCE,),
#                              {'width':width, 'height':height, 'crs':3857, 'time':('2017-03-30','2017-05-03'), 'layers':'TRUE_COLOR', 'maxcc':maxcc})
# downloader_train_images.download_data()
