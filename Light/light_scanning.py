import itertools

import cv2
import numpy as np
import pandas as pd
from itertools import combinations, product

dewarp_coords = {
        'W': (
            (304, 312),
            (344, 735),
            (1001, 710),
            (1003, 305)
        ),
        'R': (
            (305, 306),
            (343, 732),
            (1002, 707),
            (1004, 302)
        ),
        'G': (
            (303, 304),
            (343, 726),
            (1000, 702),
            (1002, 298)
        ),
        'B': (
            (301, 308),
            (341, 729),
            (999, 704),
            (1000, 302)
        ),
        'C': (
            (299, 308),
            (341, 729),
            (998, 703),
            (1000, 299)
        ),
        'M': (
            (298, 307),
            (341, 728),
            (998, 703),
            (1000, 299)
        ),
        'Y': (
            (296, 308),
            (339, 730),
            (996, 704),
            (998, 299)
        ),
}
unwarped_shape = (424, 699, 3)
paint_color_names = [
    ['Raspberry', 'Maroon', 'Red', 'Red Orange', 'Orange', 'Light Orange', 'Yellow Orange', 'Mango', 'Golden Yellow', 'Yellow'],
    ['Lemon Yellow', 'Lime Green', 'Yellow Green', 'Jade Green', 'Green', 'Pine Green', 'Teal', 'Green Blue', 'Aqua Green', 'Turquoise'],
    ['Sky Blue', 'Cerulean', 'Light Blue', 'Blue', 'Navy Blue', 'Violet (Purple)', 'Orchid', 'Mauve', 'Pale Rose', 'Pink'],
    ['Magenta', 'Bubblegum', 'Salmon', 'Peach', 'Light Brown', 'Mahogany', 'Brown', 'Dark Brown', 'Taupe', 'Sand'],
    ['Tan', 'Harvest Gold', 'Bronze Yellow', 'Metallic Gold', 'Metallic Silver', 'Gray', 'Cool Gray', 'Slate', 'Black', 'White'],
]
light_color_names = 'WRGBCMY'

def unwarped():
    image_dict = dict()
    for light_color in light_color_names:
        image = cv2.imread(f'base images/{light_color}.jpg')
        image_dict[light_color] = image

    dewarp_images(image_dict)

    return image_dict


def find_pigment_colors():
    image_dict = unwarped()

    top_left_point = (78, 58)
    bottom_right_point = (623, 360)
    x_step = (bottom_right_point[0] - top_left_point[0]) / 9
    y_step = (bottom_right_point[1] - top_left_point[1]) / 4

    paint_color_dict = dict()
    color_name_to_coord = dict()
    coord_to_color_name = dict()
    for row in range(5):
        for col in range(10):
            coord = (row, col)
            color_name = paint_color_names[row][col]
            coord_to_color_name[coord] = color_name
            color_name_to_coord[color_name] = coord

    for row in range(5):
        for col in range(10):
            paint_color = coord_to_color_name[(row, col)]
            height, width, depth = unwarped_shape
            circle_img = np.zeros((height, width), np.uint8)
            circle_center = (
                    int(top_left_point[0] + col * x_step),
                    int(top_left_point[1] + row * y_step),
            )
            cv2.circle(circle_img, circle_center, 6, 1, thickness=-1)
            for light_color, image in image_dict.items():

                masked_data = cv2.bitwise_and(image, image, mask=circle_img)

                # cv2.imshow("masked", masked_data)
                avg_color = reversed(cv2.mean(masked_data, mask=circle_img)[:-1])
                avg_color = np.array([round(rgb) for rgb in avg_color])
                paint_color_dict[(light_color, paint_color)] = avg_color

    data = []
    for (light_color, paint_color), avg_color in paint_color_dict.items():
        data.append((light_color, paint_color, avg_color))
    return pd.DataFrame(data=data, columns=['Light Color', 'Paint Color', 'Average'])


def dewarp_images(image_dict):
    # stolen from here
    # https://theailearner.com/tag/cv2-getperspectivetransform/

    for light_color in light_color_names:
        pt_A = dewarp_coords[light_color][0]
        pt_B = dewarp_coords[light_color][1]
        pt_C = dewarp_coords[light_color][2]
        pt_D = dewarp_coords[light_color][3]

        # Here, I have used L2 norm. You can use L1 also.
        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        image_dict[light_color] = cv2.warpPerspective(image_dict[light_color], M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
        image_dict[light_color] = cv2.resize(image_dict[light_color], unwarped_shape[:-1][::-1])


def find_relationships(color_data):
    assert isinstance(color_data, pd.DataFrame)
    df = color_data

    all_paint_colors = itertools.chain.from_iterable(paint_color_names)
    all_light_colors = light_color_names
    all_indices = product(all_paint_colors, all_light_colors)

    combined = combinations(all_indices, 2)
    pairs = [(paint_1, light_1, paint_2, light_2) for (paint_1, light_1), (paint_2, light_2) in combined]
    # Here's a dataframe with 4 columns of all possible combinations of light/paint
    pairwise_df = pd.DataFrame(data=pairs, columns=['Paint 1', 'Light 1', 'Paint 2', 'Light 2'])

    def create_new_columns(row):
        nonlocal df
        paint_1, light_1 = tuple(row.iloc[:2])
        paint_2, light_2 = tuple(row.iloc[2:])
        average1 = df.loc[
                  (df['Light Color'] == light_1) &
                  (df['Paint Color'] == paint_1)]['Average'].values[0]
        average2 = df.loc[
                  (df['Light Color'] == light_2) &
                  (df['Paint Color'] == paint_2)]['Average'].values[0]
        difference = average2 - average1
        distance = np.linalg.norm(average2-average1)
        return pd.Series(np.append(difference, distance))

    pairwise_df[['R diff', 'G diff', 'B diff', 'distance']] = pairwise_df.apply(create_new_columns, axis=1)
    return pairwise_df



def do_stuff():
    color_data = find_pigment_colors()
    # print(color_dict.values()) # https://chart-studio.plotly.com/create/?fid=plotly2_demo:437#/
    try:
        relationships = pd.read_pickle('./color_relationships.pkl')
    except:
        relationships = find_relationships(color_data)
        relationships = relationships.to_pickle('./color_relationships.pkl')
    print(relationships)



if __name__ == '__main__':
    do_stuff()
