
import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
import imutils

import cv2

from align_dlib import AlignDlib

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def main(input_dir, output_dir, crop_dim):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_types =  ('**/*.jpg','**/*.JPG','*.JPG','*.jpg')
    image_paths = []
    for file_type in file_types:
        image_paths = image_paths + (glob.glob(os.path.join(input_dir, file_type)))


    for index, image_path in enumerate(image_paths):

        output_path = os.path.join(output_dir,os.path.splitext(os.path.basename(image_path))[0])
        pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))


def preprocess_image(input_path, output_path, crop_dim):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    image = None
    aligned_image = None

    image = _buffer_image(input_path)



    if image is not None:
        bb_array = align_dlib.getAllFaceBoundingBoxes(image)

        for key,bb in enumerate(bb_array):
            aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
            if aligned is not None:
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                if aligned is not None:
                    logger.debug('Writing processed file: {}'.format(output_path))
                    cv2.imwrite(output_path+str(key)+'.jpg', aligned)
                else:
                    logger.warning("Skipping filename: {}".format(input_path))
    else:
        raise IOError('Error buffering image: {}'.format(input_path))


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = imutils.resize(image, width=2000)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=180, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.crop_dim)
