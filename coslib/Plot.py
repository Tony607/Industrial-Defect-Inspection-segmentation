import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from .Transform import get_mask_seg, get_mask_seg_ellipse, get_mask_seg_polygon
from .Utils import get_coordinates

def plot_ellipse(raw_img, coordinates):
    """plot image with ellipse contour


    """

    img_copy = raw_img.copy()

    # return img_copy

    return_img = cv2.ellipse(img_copy,
                             (coordinates['x'], coordinates['y']),
                             (coordinates['major_axis'], coordinates['minor_axis']),
                             coordinates['angle'] * 90,
                             # -90,
                             0,
                             360,
                             (0, 0, 0),
                             2)

    plt.imshow(return_img, cmap='gray')
    print('angle: {}'.format(coordinates['angle']))
    print('angle(degree): {}'.format(90 * coordinates['angle']))


    return return_img



def plot_bbox(path_to_image, xml):
    '''plot bbox on raw image

    Args:
        raw_img (numpy array):
        coordinates (dict):

    Return:

    '''
    img = mpimg.imread(path_to_image)

    if xml:
        path_to_label = path_to_image.replace('.bmp', '.xml')
        # print(path_to_label)
        coordinates_object = get_coordinates(path_to_label, xml=True)
    else:
        path_to_label = path_to_image.replace('.bmp', '.txt')
        # print(path_to_label)
        coordinates_object = get_coordinates(path_to_label, xml=False)
        # return coordinates_object
    # return coordinates_object

    img_copy = img.copy()
    if type(coordinates_object) is list:
        print('{} bounding boxes'.format(len(coordinates_object)))
        for i in range(len(coordinates_object)):
            if xml:
                coordinates = coordinates_object[i]['bndbox']

                print('Defect type: {}'.format(coordinates_object[i]['name']))
                print('xmin: {} ymin: {} xmax: {} ymax:{}'.format(int(coordinates['xmin']),
                                                                                                        int(coordinates['ymin']),
                                                                                                        int(coordinates['xmax']),
                                                                                                        int(coordinates['ymax'])))

                return_img = cv2.rectangle(img_copy, (int(coordinates['xmin']), int(coordinates['ymin'])),
                                                                             (int(coordinates['xmax']), int(coordinates['ymax'])),
                                                                             (255, 0, 0),5)


            else:
                # return coordinates_object
                # print(coordinates_object)
                print('Defect type: {}'.format(coordinates_object[i].split()[0]))
                print('xmin: {} ymin: {} xmax: {} ymax:{}'.format(int(coordinates_object[i].split()[4]),
                                                                                                        int(coordinates_object[i].split()[5]),
                                                                                                        int(coordinates_object[i].split()[6]),
                                                                                                        int(coordinates_object[i].split()[7])))

                return_img = cv2.rectangle(img_copy, (int(coordinates_object[i].split()[4]),
                                                                             int(coordinates_object[i].split()[5])),
                                                                             (int(coordinates_object[i].split()[6]),
                                                                             int(coordinates_object[i].split()[7])),
                                                                             (255, 0, 0), 5)
    else:
        coordinates = coordinates_object['bndbox']
        print('1 bounding box')
        print('Defect type: {}'.format(coordinates_object['name']))
        print('xmin: {} ymin: {} xmax: {} ymax:{}'.format(int(coordinates['xmin']),
                                                          int(coordinates['ymin']),
                                                          int(coordinates['xmax']),
                                                          int(coordinates['ymax'])))
        return_img = cv2.rectangle(img_copy, (int(coordinates['xmin']), int(coordinates['ymin'])),
                                                                         (int(coordinates['xmax']), int(coordinates['ymax'])),
                                                                         (255, 0, 0),5)
    plt.imshow(return_img, cmap='gray')

    return return_img


def plot_bbox_seg_test(path_to_img, xml=True):
    """plot bbox and seg

    Args:
      path_to_img (str): path to image
      xml (boolean): specified label data, xml or txt

    Return: None

    Notes:
    """
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    test = plot_bbox(path_to_img, xml=True)
    plt.subplot(1, 2, 2)
    img_mask = get_mask_seg(path_to_img, xml=True)
    plt.imshow(img_mask, cmap='gray')


def plot_ellipse_seg_test(path_to_img):

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)

    plt.imshow(mpimg.imread(path_to_img), cmap='gray')
    plt.subplot(1, 2, 2)
    mask = get_mask_seg_ellipse(path_to_img)
    plt.imshow(mask, cmap='gray')
    
def plot_polygon_seg_test(path_to_img, gray=True):

    plt.figure(figsize=(30, 20))
    plt.subplot(1, 3, 1)
    
    raw_img = mpimg.imread(path_to_img)
    if gray:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
    plt.imshow(raw_img)
    plt.subplot(1, 3, 2)
    mask = get_mask_seg_polygon(path_to_img)
    im_mask = np.array(255*mask, dtype=np.uint8)
    rgb_mask = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
    rgb_mask[:,:,1:3] = 0*rgb_mask[:,:,1:2]
    
    plt.imshow(mask, cmap='gray')
    
    img_mask = cv2.addWeighted(raw_img, 1, rgb_mask, 0.8, 0)
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_mask)