import matplotlib.image as mpimg

import glob
import os
import xmltodict
import numpy as np
from .Utils import get_coordinates, load_coordinates
import cv2
import json


def xml_to_kitti(path_to_xml):
    """transform xml format to xml format

    Arguments:
        path_to_xml (string):

    Notes:

    Return:

    """

    # defect_type = os.path.dirname(path_to_xml).split('/')[-1]

    for xml_file in glob.glob('{}/*.xml'.format(path_to_xml)):
        with open(xml_file) as f:
            label_in_xml = xmltodict.parse(f.read())['annotation']['object']

            root_name = os.path.splitext(os.path.basename(xml_file))[0] + '.txt'
            path_to_output = os.path.join(path_to_xml, root_name)
            f_out = open(path_to_output, 'w')

            # print('writing file: {}'.format(root_name))
            if type(label_in_xml) is list:
                # return label_in_xml
                for bbox_object in label_in_xml:
                    bbox = bbox_object['bndbox']
                    defect_type = bbox_object['name']
                    f_out.write('{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0'.format(defect_type,
                                                                            bbox['xmin'],
                                                                            bbox['ymin'],
                                                                            bbox['xmax'],
                                                                            bbox['ymax']))
                    f_out.write('\n')
            else:
                bbox = label_in_xml['bndbox']
                defect_type = label_in_xml['name']
                f_out.write('{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0'.format(defect_type,
                                                                        bbox['xmin'],
                                                                        bbox['ymin'],
                                                                        bbox['xmax'],
                                                                        bbox['ymax']))
            f_out.close()


def get_mask_seg(path_to_img, xml=True):
    '''create image for segmentation application

    Arg:
        img (numpy array): original grayscale or color rgb image
        coord_min (tuple): (xmin, ymin)
        coord_max (tuple): (xmax, ymax)

    Return:

    Notes:
    '''

    img = mpimg.imread(path_to_img)

    img_mask = np.zeros_like(img[:, :])

    if xml:
        path_to_xml = path_to_img.replace('bmp', 'xml')
        # print('path_to_xml: {}'.format(path_to_xml))
        coor_obj = get_coordinates(path_to_xml, xml)

        # if there are more than 1 bounding boxes
        if type(coor_obj) is list:
            for i in range(len(coor_obj)):
                xmin = int(coor_obj[i]['bndbox']['xmin'])
                ymin = int(coor_obj[i]['bndbox']['ymin'])
                xmax = int(coor_obj[i]['bndbox']['xmax'])
                ymax = int(coor_obj[i]['bndbox']['ymax'])

                img_mask[ymin:ymax, xmin:xmax] = 1.
        else:
            # there is only one bounding box
            # print('test')
            xmin = int(coor_obj['bndbox']['xmin'])
            ymin = int(coor_obj['bndbox']['ymin'])
            xmax = int(coor_obj['bndbox']['xmax'])
            ymax = int(coor_obj['bndbox']['ymax'])

            img_mask[ymin:ymax, xmin:xmax] = 1.

    return img_mask


def get_mask_seg_ellipse(path_to_img):
    """
    """

    # get the image

    img = mpimg.imread(path_to_img)
    basename = os.path.basename(path_to_img)

    # filename_index, e.g. filename = 1.png
    # filename_index = 1, for extracting coordinates
    filename_index = int(os.path.splitext(basename)[0]) - 1
    # print(filename_index)

    path_to_coordinates = path_to_img.replace(basename, 'labels.txt')
    coordinates = load_coordinates(path_to_coordinates)

    mask = np.zeros_like(img)
    mask = cv2.ellipse(mask, 
                       (int(coordinates[filename_index]['x']), int(coordinates[filename_index]['y'])),
                       (int(coordinates[filename_index]['major_axis']), int(coordinates[filename_index]['minor_axis'])),
                       (coordinates[filename_index]['angle'] / 4.7) * 270,
                       0, 
                       360, 
                       (255, 255, 255), 
                       -1)

    mask[mask > 0] = 1.

    # print(coordinates[filename_index]['angle'])

    return mask

def get_mask_seg_polygon(path_to_img, gray=True):
    img = mpimg.imread(path_to_img)
    
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    basename = os.path.basename(path_to_img)
    
    path_to_json = path_to_img.replace('bmp', 'json')
    
    # print(path_to_json)
    
    with open(path_to_json) as json_file:
        json_get = json.load(json_file)
        pts_list = [np.array(pts['points'], np.int32) for pts in json_get['shapes']]
    
    
    # return np.array(pts, np.int32)
    
    mask = np.zeros_like(img)
    
    # rgb_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
    # return mask
    mask = cv2.fillPoly(mask, pts_list, (255, 255, 255))
    
    mask[mask > 0] = 1.
    
    return mask
