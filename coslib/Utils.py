import xmltodict
import cv2
import matplotlib.image as mpimg
import os
import shutil


def get_coordinates(path_to_label, xml):
    """
    """
    with open(path_to_label, encoding='utf-8') as f:
        if xml:
            label_xml = xmltodict.parse(f.read())

            # print(type(label_xml))
            # print(label_xml)

            coordinates_object = label_xml['annotation']['object']
        else:
            label_txt = f.read()
            coordinates_object = label_txt.strip().split('\n')

    return coordinates_object


def load_coordinates(path_to_coor):
    '''
    '''

    coord_dict = {}
    coord_dict_all = {}
    with open(path_to_coor) as f:
        coordinates = f.read().split('\n')
        for coord in coordinates:
            # print(len(coord.split('\t')))
            if len(coord.split('\t')) == 6:
                coord_dict = {}
                coord_split = coord.split('\t')
                # print(coord_split)
                # print('\n')
                coord_dict['major_axis'] = round(float(coord_split[1]))
                coord_dict['minor_axis'] = round(float(coord_split[2]))
                coord_dict['angle'] = float(coord_split[3])
                coord_dict['x'] = round(float(coord_split[4]))
                coord_dict['y'] = round(float(coord_split[5]))
                index = int(coord_split[0]) - 1
                coord_dict_all[index] = coord_dict

    return coord_dict_all


def get_image(path_to_image):

    return mpimg.imread(path_to_image)


def separate_xml_files(path_to_images):
    '''
    '''

    file_basenames = [os.path.splitext(x)[0] for x in os.listdir(path_to_images) if x.endswith('xml')]
    image_directory = path_to_images.split('/')[-1]
    new_image_directory = image_directory + '_xml'
    print('image_directory {}'.format(image_directory))
    print('new image directory {}'.format(new_image_directory))
    # directory_newname = os.path.basename(path_to_images) + '_xml'
    # print(directory_newname)
    # os.makedirs(directory_newname)
    for basename in file_basenames:
        # print(basename)
        original_img = os.path.join(path_to_images, basename)
        remove_img = original_img.replace(image_directory, new_image_directory)
        print(remove_img)
        shutil.copy('{}.bmp'.format(original_img), '{}.bmp'.format(remove_img))
        shutil.copy('{}.xml'.format(original_img), '{}.xml'.format(remove_img))
