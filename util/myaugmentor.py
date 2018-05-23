from Augmentor import Pipeline
from tqdm import tqdm
import os
from datetime import datetime

class MyPipeline(Pipeline):

    def ground_truth(self, ground_truth_directory):
        """
        Modification of the original method to conform to our mask naming convention.
        
        Specifies a directory containing corresponding images that
        constitute respective ground truth images for the images
        in the current pipeline.

        This function will search the directory specified by
        :attr:`ground_truth_directory` and will associate each ground truth
        image with the images in the pipeline by file name.
    
        The relationship between image and ground truth filenames is the following:
        img filename: 'img/tulip_<PATCH_ID>_wms_<DATE>_<SENTINELHUB_LAYER>.png'
        ground truth filename: 'mask/tulip_<PATCH_ID>_geopedia_<GEOPEDIA_LAYER>.png' 
        
        Typically used to specify a set of ground truth or gold standard
        images that should be augmented alongside the original images
        of a dataset, such as image masks or semantic segmentation ground
        truth images.

        :param ground_truth_directory: A directory containing the
         ground truth images that correspond to the images in the
         current pipeline.
        :type ground_truth_directory: String
        :return: None.
        """
        geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905'}
        num_of_ground_truth_images_added = 0

        # Progress bar
        progress_bar = tqdm(total=len(self.augmentor_images), desc="Processing", unit=' Images', leave=False)

        for augmentor_image_idx in range(len(self.augmentor_images)):
            filename = os.path.basename(self.augmentor_images[augmentor_image_idx].image_file_name)
            patch_id = filename.split('_')[1]
            year = datetime.strptime(filename.split('_')[3], "%Y%m%d-%H%M%S").year
            mask_fn = 'tulip_{}_geopedia_{}.png'.format(patch_id, geopedia_layers['tulip_field_{}'.format(year)])
            ground_truth_image = os.path.join(ground_truth_directory,
                                              mask_fn)
            if os.path.isfile(ground_truth_image):
                self.augmentor_images[augmentor_image_idx].ground_truth = ground_truth_image
                num_of_ground_truth_images_added += 1
                progress_bar.update(1)

        progress_bar.close()

        # May not be required after all, check later.
        if num_of_ground_truth_images_added != 0:
            self.process_ground_truth_images = True

        print("%s ground truth image(s) found." % num_of_ground_truth_images_added)



