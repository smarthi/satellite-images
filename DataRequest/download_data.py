'''
Script to download the blooming tulips dataset from sentinelhub
'''
from gather_data import PolygonSlidingWindow, GeoJsonSaver, BatchDownloader
from DataRequest import TulipFieldRequest, S2Request

WMS_INSTANCE = '71513b0b-264d-494a-b8c4-c3c36433db28'
geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}

root_dir = '../data/tulips/bloom/'

# Params
width = 256
height = 256

# 2016
poly_sw_cb = PolygonSlidingWindow(box_width=2560, box_height=2560, stride_x=2304, stride_y=2304)
poly_sw_cb.set_mode('train')
poly_sw_cb.load_polygons_from_folder('../data/tulips/poly/16/')

# Download labels
downl_labels = BatchDownloader(root_dir + 'masks/', poly_sw_cb, TulipFieldRequest, (),
							 {'width':width, 'height':height, 'crs':3857, 'layer':geopedia_layers['tulip_field_2016']})
downl_labels.download_data()

# Download images
downl_imgs = BatchDownloader(root_dir + '16/', poly_sw_cb,
                             S2Request, (WMS_INSTANCE,),
                             {'width':width, 'height':height, 'crs':3857,
                              'time':('2016-04-30','2016-05-13'), 'layers':'TRUE_COLOR', 'maxcc':0.7})
downl_imgs.download_data()

# 2017
poly_sw_cb = PolygonSlidingWindow(box_width=2560, box_height=2560, stride_x=2304, stride_y=2304)
poly_sw_cb.set_mode('train')
poly_sw_cb.load_polygons_from_folder('../data/tulips/poly/17/')

# Download labels
downl_labels = BatchDownloader(root_dir + 'masks/', poly_sw_cb, TulipFieldRequest, (),
							 {'width':width, 'height':height, 'crs':3857, 'layer':geopedia_layers['tulip_field_2017']})
downl_labels.download_data()

# Download images
downl_imgs = BatchDownloader(root_dir + '17/', poly_sw_cb,
                             S2Request, (WMS_INSTANCE,),
                             {'width':width, 'height':height, 'crs':3857, 
                             'time':('2017-04-10','2017-05-15'), 'layers':'TRUE_COLOR', 'maxcc':0.7})
downl_imgs.download_data()
