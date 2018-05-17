import sys
from DataRequest import TulipFieldRequest, S2Request
from gather_data import BatchDownloader, PolygonSlidingWindow

WMS_INSTANCE = '71513b0b-264d-494a-b8c4-c3c36433db28'
sentinel_hub_wms='https://services.sentinel-hub.com/ogc/wms/'+WMS_INSTANCE
geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}
s2_wms_layers = ['TRUE_COLOR', 'ALL_BANDS']

img_shape = (256, 256)

poly_sw_cb = PolygonSlidingWindow(box_width=img_shape[0]*10, box_height=img_shape[1]*10,
 								stride_x=img_shape[0]*9, stride_y=img_shape[1]*9)
poly_sw_cb.set_mode('train')
poly_sw_cb.load_polygons_from_folder('../data/tulips/poly/16/')


root_dir = '../data/tulips/bloom/16/'


poly_sw_cb.set_mode('train')
downl_labels = BatchDownloader(root_dir, poly_sw_cb, TulipFieldRequest, (), 
								{'width':img_shape[0], 'height':img_shape[1], 'crs':3857, 
								'layer':geopedia_layers['tulip_field_2016']})
downl_labels.download_data()

downl_images = BatchDownloader(root_dir, poly_sw_cb,
                             S2Request, (WMS_INSTANCE,),
                             {'width':img_shape[0], 'height':img_shape[1], 'crs':3857,
                             'time':('2016-04-30','2016-05-13'), 'layers':'TRUE_COLOR', 'maxcc':0.7})
downl_images.download_data()