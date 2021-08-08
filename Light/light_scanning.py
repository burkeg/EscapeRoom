import itertools
import pickle
import cv2
import numpy as np
import pandas as pd
from itertools import combinations, product
import oapackage

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
    ['Raspberry', 'Maroon', 'Red', 'Red_Orange', 'Orange', 'Light_Orange', 'Yellow_Orange', 'Mango', 'Golden_Yellow', 'Yellow'],
    ['Lemon_Yellow', 'Lime_Green', 'Yellow_Green', 'Jade_Green', 'Green', 'Pine_Green', 'Teal', 'Green_Blue', 'Aqua_Green', 'Turquoise'],
    ['Sky_Blue', 'Cerulean', 'Light_Blue', 'Blue', 'Navy_Blue', 'Violet_or_Purple', 'Orchid', 'Mauve', 'Pale_Rose', 'Pink'],
    ['Magenta', 'Bubblegum', 'Salmon', 'Peach', 'Light_Brown', 'Mahogany', 'Brown', 'Dark_Brown', 'Taupe', 'Sand'],
    ['Tan', 'Harvest_Gold', 'Bronze_Yellow', 'Metallic_Gold', 'Metallic_Silver', 'Gray', 'Cool_Gray', 'Slate', 'Black', 'White'],
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


def find_relationships():
    color_data = find_pigment_colors()

    try:
        return pd.read_pickle('./color_relationships.pkl')
    except:
        print('Recalculating color_relationships')
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
    pairwise_df.to_pickle('./color_relationships.pkl')
    return pairwise_df

def analyze():
    relationships = find_relationships()
    light_color_relationships = dict()
    for light_color in light_color_names:
        light_color_only = relationships.loc[
               (relationships['Light 1'] == relationships['Light 2']) &
               (relationships['Light 1'] == light_color)]
        assert isinstance(light_color_only, pd.DataFrame)
        light_color_only = light_color_only.drop(['R diff', 'G diff', 'B diff'], axis=1)
        light_color_only = light_color_only.sort_values(by='distance', ignore_index=True)
        light_color_relationships[light_color] = light_color_only

    # light_color_relationships is a dictionary with 1 entry for each light color
    # Each value is a sorted list showing which paints look the most similar to each other in that light
    # For each combination of lights, lets see which colors are the most similar in 1 lighting but
    # maximally different in another lighting
    def create_new_columns(row):
        nonlocal light_2_diffs
        df = light_2_diffs
        paint_1, paint_2 = (row.iloc[0], row.iloc[2])
        my_index = row.name
        other_index = df.index[
                  (df['Paint 1'] == paint_1) &
                  (df['Paint 2'] == paint_2)][0]
        my_difference = row['distance']
        other_difference = df.loc[
                  (df['Paint 1'] == paint_1) &
                  (df['Paint 2'] == paint_2)]['distance'].values[0]
        # print(my_index, other_index, my_difference, other_difference)
        index_difference = abs(other_index - my_index)
        color_difference = abs(other_difference - my_difference)
        same_in_light = df['Light 1'].values[0] if my_difference > other_difference else row['Light 1']
        threshold = 256
        min_difference = min(my_difference, other_difference)

        # for example if 'I' am white light and 'other' is red light, this function is called once for
        # each pair of colors. Let's say we're comparing aqua green and black.
        # my_index is how far apart in rank aqua green and black were in RGB distance in white light.
        # other_index is how far apart in rank aqua green and black were in RGB distance in red light.
        # my_difference is how far apart in RGB distance aqua green and black are in white light.
        # other_difference is how far apart in RGB distance aqua green and black are in red light.
        # This adds a column for the following:
        # index_difference: higher when my_index and other_index are far apart.
        # color_difference: higher when my_difference and other_difference are far apart.
        # same_in_light: was it white or red light that made aqua green and black look more similar?
        # min_difference: for the lighting where aqua green and black were more similar, how similar exactly?
        if min_difference > threshold:
            return pd.Series([np.NaN, np.NaN, np.NaN, np.NaN])
        else:
            return pd.Series([index_difference, color_difference, same_in_light, min_difference])

    try:
        with open('paired_light_data.pkl', 'rb') as handle:
            paired_light_data = pickle.load(handle)
    except:
        paired_light_data = dict()
        print('Recalculating paired_light_data')
        for light_1, light_2 in combinations(light_color_names, 2):
            light_1_diffs = light_color_relationships[light_1].sort_values(by=['Paint 1', 'Light 1', 'Paint 2', 'Light 2'])
            light_2_diffs = light_color_relationships[light_2].sort_values(by=['Paint 1', 'Light 1', 'Paint 2', 'Light 2'])
            assert isinstance(light_1_diffs, pd.DataFrame)
            light_1_diffs[['index difference', 'color difference', 'light where same', 'min difference']] = light_1_diffs.apply(create_new_columns, axis=1)
            paired_light_data[(light_1, light_2)] = light_1_diffs.drop(['Light 1', 'Light 2', 'distance'], axis=1)
            paired_light_data[(light_1, light_2)] = paired_light_data[(light_1, light_2)].sort_values(
                by=['light where same', 'color difference'],
                ignore_index=True,
                ascending=False)
            print(light_1, light_2)

        with open('paired_light_data.pkl', 'wb') as handle:
            pickle.dump(paired_light_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(paired_light_data)

    print()

    # combined_paired_light_data = pd.concat(paired_light_data.values(), keys=paired_light_data.keys())
    # combined_paired_light_data.reset_index(inplace=True)
    # combined_paired_light_data = combined_paired_light_data.drop(['level_2'], axis=1)
    # combined_paired_light_data.columns = ['Light 1', 'Light 2'] + list(combined_paired_light_data.columns)[2:]
    return paired_light_data

def calc_pareto(df):
    # https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html
    df.reset_index(inplace=True, drop=True)
    df_filtered = df.drop(['Paint 1', 'Paint 2', 'index difference', 'light where same'], axis=1)
    datapoints = df_filtered.to_numpy().transpose()
    min_distance_max = np.max(datapoints, 1)[1]
    datapoints[1, :] = min_distance_max - datapoints[1, :]

    for ii in range(0, datapoints.shape[1]):
        w = datapoints[:, ii]
        fac = .6 + .4 * np.linalg.norm(w)
        datapoints[:, ii] = (1 / fac) * w

    pareto = oapackage.ParetoDoubleLong()

    for ii in range(0, datapoints.shape[1]):
        w = oapackage.doubleVector((datapoints[0, ii], datapoints[1, ii]))
        pareto.addvalue(w, ii)

    lst = pareto.allindices()  # the indices of the Pareto optimal designs

    return df.loc(axis=0)[lst]

def get_all_pareto_fronts(paired_light_data):
    paired_light_data = analyze()
    pareto_fronts = dict() # for each combo of light colors, show the pareto front of optimal colors for each light

    for key in paired_light_data.keys():
        # l_color, r_color = key
        assert isinstance(paired_light_data[key], pd.DataFrame)
        l_color_data = paired_light_data[key].loc[
            (paired_light_data[key]['light where same'] == key[0])]
        r_color_data = paired_light_data[key].loc[
            (paired_light_data[key]['light where same'] == key[1])]

        l_color_data = calc_pareto(l_color_data)
        r_color_data = calc_pareto(r_color_data)
        pareto_fronts[key] = (l_color_data, r_color_data)
    return pareto_fronts

def do_stuff():
    # print(relationships)
    analyze()



if __name__ == '__main__':
    do_stuff()
