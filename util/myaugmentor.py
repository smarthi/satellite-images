import os
import uuid
import random

from Augmentor import Pipeline
from datetime import datetime
from tqdm import tqdm
from PIL import Image

class MyPipeline(Pipeline):

    def ground_truth(self, ground_truth_directory):
        """
        Modification of the original method to conform to our mask naming convention.
        ------
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
        num_of_ground_truth_images_added = 0
        geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905'}

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


    def _execute(self, augmentor_image, save_to_disk=True, multi_threaded=True):
        """
        Modification so that saved images also follow our naming convention.
        ------
        Private method. Used to pass an image through the current pipeline,
        and return the augmented image.

        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """

        images = []
        geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905'}

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        if augmentor_image.ground_truth is not None:
            if isinstance(augmentor_image.ground_truth, list):
                for image in augmentor_image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        if save_to_disk:
            file_name = str(uuid.uuid4())
            basename_split = os.path.basename(augmentor_image.image_path).split('.')[0].split('_')
            basename_split[1] = file_name
            year = datetime.strptime(basename_split[3], "%Y%m%d-%H%M%S").year
            try:
                for i in range(len(images)):
                    if i == 0:
                        save_name = '_'.join(basename_split) + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    else:
                        basename_split[2] = 'geopedia'
                        basename_split[3] = geopedia_layers['tulip_field_{}'.format(year)]
                        save_name = '_'.join(basename_split[:4]) + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

        # TODO: Fix this really strange behaviour.
        # As a work around, we can pass the same back and basically
        # ignore the multi_threaded parameter completely for now.
        # if multi_threaded:
        #   return os.path.basename(augmentor_image.image_path)
        # else:
        #   return images[0]  # Here we return only the first image for the generators.

        return images[0]
