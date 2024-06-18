import numpy as np
from PIL import Image
import scipy.signal
from scipy import ndimage
import random

np.seterr(all='ignore')


sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])


def harris_corner_responses(input_im, threshold, mask=None):
    """
    1. Calculate the Harris corner responses of input_im.
    2. Return the coordinates of corners which are above the provided threshold and are local maximum points.
    :param input_im: 2D grayscale numpy array.
    :param threshold: Threshold above which corners will be returned.
    :param mask: A mask of same dimensions as input_im, used to optimize the calculation of the corners
    by disregarding pixels whose value is zero.
    :return: np.argwhere of the corners that were found.
    """

    # Blur the image with a Gaussian filter
    im = ndimage.gaussian_filter(input_im, 2)

    # Compute derivatives used for calculation of corners
    Ix = scipy.signal.convolve(im, sobel_x, mode="same")
    Ix_squared = Ix ** 2
    Iy = scipy.signal.convolve(im, sobel_y, mode="same")
    Iy_squared = Iy ** 2
    IxIy = Ix * Iy

    # This array contains the unique matrix of each pixel, corresponding to those being summed while calculating M for
    # a certain pixel
    Ms_before_summation = np.zeros(shape=im.shape + (2, 2))
    Ms_before_summation[:, :, 0, 0] = Ix_squared
    Ms_before_summation[:, :, 0, 1] = IxIy
    Ms_before_summation[:, :, 1, 0] = IxIy
    Ms_before_summation[:, :, 1, 1] = Iy_squared

    # Each 2x2 array here is now of this shape:
    # np.array([[Ix_squared[y,x],      IxIy[y,x]],
    #           [IxIy[y,x],     Iy_squared[y,x]]])

    responses = np.zeros_like(im).astype(int)
    # We are using 3x3 windows, so we don't calculate score for the image borders (where the border will be the
    # middle pixel). We split this into two cases - one where alpha_channel is not None and one where it is. The
    # reason for this is to avoid calling the if condition 1920*910=1747200 times.

    if mask is not None:
        for y in range(1, responses.shape[0] - 1):
            for x in range(1, responses.shape[1] - 1):
                if (mask[y, x] == 0):  # This "if" is the only addition to the "else" clause below
                    continue
                M = np.sum(Ms_before_summation[y - 1:y + 2, x - 1:x + 2, ...], axis=(0, 1))
                try:
                    responses[y, x] = np.linalg.det(M) // np.trace(M)
                except:
                    responses[y, x] = 0
    else:
        for y in range(1, responses.shape[0] - 1):
            for x in range(1, responses.shape[1] - 1):
                M = np.sum(Ms_before_summation[y - 1:y + 2, x - 1:x + 2, ...], axis=(0, 1))
                try:
                    responses[y, x] = np.linalg.det(M) // np.trace(M)
                except:
                    responses[y, x] = 0

    # "responses" now has the Harris corner responses. We will filter them out by keeping only local maxima.

    # Smooth responses
    responses = ndimage.gaussian_filter(responses, 1)

    # Differentiate responses
    responses_dx = scipy.signal.convolve(responses, sobel_x, mode="same")
    responses_dy = scipy.signal.convolve(responses, sobel_y, mode="same")

    # Local maximum points of "responses" correspond to a zero crossing of the derivative from positive to negative.
    # We will determine where this crossing takes place by running np.diff(np.sign(responses_dx), axis=1) and checking
    # for negative values (like checking the second derivative). This will also be applied to the other axis as well.
    # Why use np.diff(np.sign(responses_dx), axis=1):
    # If the derivative was positive, np.sign will return 1 in its place. np.diff subtracts a cell from the cell before,
    # so if the derivative value in two cells was +, the subtraction will result in 0.
    # If there was a change of + to -, the result will be negative, indicating a local maximum, the thing we are looking
    # for.
    # The diff function shrinks the array by a single column/row (depending on the axis). So we concatenate a zero to
    # make it the same size again.
    maxima_x = np.concatenate((np.zeros((responses.shape[0], 1)), np.diff(np.sign(responses_dx), axis=1) < 0), axis=1)
    maxima_y = np.concatenate((np.zeros((1, responses.shape[1])), np.diff(np.sign(responses_dy), axis=0) < 0), axis=0)

    # We take maximum points which are maximal both in the x and y directions.
    maxima = maxima_x * maxima_y

    # Return the responses which were deemed maximum points.
    responses = maxima * responses

    # Return the coordinates of those points.
    return np.argwhere(responses > threshold)


def get_descriptors(input_im, coords):
    """
    Given an image and a list of coordinates, return a dictionary with coordinates as keys and descriptors as values.
    Descriptors are of size 21x21 pixels.
    :param input_im: The input image which the coordinate points at.
    :param coords: List of coordinates whose descriptors we need to be compute.
    :return: A dictionary of coordinate: descriptor.
    """

    descriptors = {}

    # Blur the image and subsample it.
    reduced_image = ndimage.gaussian_filter(input_im, 3)[1::2, 1::2]

    # Calculate the gradients of every pixel in reduced_image (and blur afterward)
    Ix = scipy.signal.convolve(reduced_image, sobel_x, mode="same")
    Iy = scipy.signal.convolve(reduced_image, sobel_y, mode="same")
    gradient_arr = ndimage.gaussian_filter(np.degrees(np.arctan2(Iy, Ix)), 1)

    # Find the descriptor for each coordinate
    for coord in coords:
        coordinate_reduced = coord // 2
        radius = 10
        mask_padding = radius
        gradient = np.mean(gradient_arr[coordinate_reduced[0] - radius:coordinate_reduced[0] + radius + 1,
                           coordinate_reduced[1] - radius:coordinate_reduced[1] + radius + 1])

        # Rotation generates black triangles where there is no data. So we rotate a bigger patch and then crop it.

        patch_large = reduced_image[
                      coordinate_reduced[0] - radius - mask_padding:coordinate_reduced[0] + radius + 1 + mask_padding,
                      coordinate_reduced[1] - radius - mask_padding:coordinate_reduced[1] + radius + 1 + mask_padding]
        rotated_patch_large = ndimage.gaussian_filter(ndimage.rotate(patch_large, angle=-gradient, reshape=False), 2)
        rotated_patch = rotated_patch_large[mask_padding:-mask_padding, mask_padding:-mask_padding]

        if 0 not in coordinate_reduced and rotated_patch.shape == (21, 21):
            descriptors[tuple(coord)] = rotated_patch

    return descriptors


def find_matching_pairs(descriptors_1, descriptors_2):
    """
    Find matching descriptors from descriptors_1 and descriptors_2.
    Point B matches point A if the ratio dist(A,B)/dist(A,C) is the smallest, where C can be any third point and dist is
    the Euclidean distance of the descriptors. Note that pairs are unique.
    :param descriptors_1: Descriptors as returned from the function get_descriptors()
    :param descriptors_2: Descriptors as returned from the function get_descriptors()
    :return: A list of 2 element lists, each of which is a pairing between a point in the first image and a point in the
     second, sorted by score.
    """
    # Calculate the distance between each pair of descriptors
    descriptors_2_keys = list(descriptors_2.keys())
    matches = {}
    distances = np.zeros(len(descriptors_2))
    for key1, mat1 in descriptors_1.items():
        for i, mat2 in enumerate(descriptors_2.values()):
            distances[i] = np.sum((mat1 - mat2) ** 2)

        NN1, NN2 = np.argsort(distances)[:2]
        matching_coordinate = descriptors_2_keys[NN1]
        score = distances[NN1] / distances[NN2]

        if matching_coordinate in matches:
            if score < matches[matching_coordinate][1]:
                matches[matching_coordinate] = [key1, score]
        else:
            matches[matching_coordinate] = [key1, score]

    # Return sorted distances by score. Pairs are unique

    result = {coord: match for coord, match in matches.items() if match[1] < 1}
    sorted_result = sorted(result.items(), key=lambda x: x[1][1])
    return [[row[0], row[1][0]] for row in sorted_result]


def ransac(pairs, p=0.99, epsilon=4):
    """
    Return transformation based on 4 pairs of points which satisfies the most pairs of points (has the greatest number
    of inliers).
    :param pairs: A list like this: [[(150, 100), (33, 10)], [(123, 56), (89, 101)], ... ] the first point in every pair
    is a point in image1, the second is its guess of a match in image2.
    :param p: the probability we choose 4 inliers in some iterations.
    :param epsilon: the distance we permit a point after transformation be in comparison to the match of the original
    point.
    :return: A tuple: A 3x3 matrix which represents the transformation that satisfied the most pairs, the pairs which
    were satisfied.
    """

    def get_transformation(pairs):
        """
        Given 4 pairs of coordinates, calculate the projective transformation matrix satisfying these four pairs.
        :param pairs: 4 pairs of points.
        :return: A 3x3 matrix of a projective transformation
        """
        if len(pairs) != 4:
            raise ("You must supply 4 pairs to calculate the transformation.")
        equations = None
        b = None
        for i, pair in enumerate(pairs):
            u = pair[0]
            v = pair[1]
            rows_to_add = np.array([[u[0], u[1], 1, 0, 0, 0, -u[0] * v[0], -u[1] * v[0]],
                                    [0, 0, 0, u[0], u[1], 1, -u[0] * v[1], -u[1] * v[1]]])
            b_to_add = np.array([[v[0]], [v[1]]])
            if i == 0:
                equations = rows_to_add
                b = b_to_add
            else:
                equations = np.vstack((equations, rows_to_add))
                b = np.vstack((b, b_to_add))

        try:
            solution = np.linalg.solve(equations, b)
            A = np.reshape(np.append(solution, 1), (3, 3))
            return A
        except:
            return None

    epsilon_squared = epsilon ** 2

    best_model = None
    best_inliers = []

    num_of_iterations = np.inf
    cur_iteration = 0
    while cur_iteration < num_of_iterations:

        # 1. Randomly choose 4 indices (corresponding to 4 pairs)

        indices = random.sample(range(len(pairs)), 4)
        pairs_cur = [pairs[i] for i in indices]

        # 2. Create the transformation given these pairs

        A = get_transformation(pairs_cur)
        if A is None:
            continue

        # 3. Find the pairs which are inliers using vectorized operations

        p1 = np.array([point1 for point1, _ in pairs])
        p2 = np.array([point2 for _, point2 in pairs])

        h_point1 = np.column_stack((p1, np.ones(len(p1))))
        h_A_times_p1 = np.dot(A, h_point1.T).T
        A_times_p1 = (h_A_times_p1 / h_A_times_p1[:, 2][:, np.newaxis])[:, :2]

        distances_squared = np.sum((A_times_p1 - p2) ** 2, axis=1)
        test_inliers = [tuple(pair) for pair, distance_squared in zip(pairs, distances_squared) if
                        distance_squared < epsilon_squared]

        # 4. Keep the model with the largest set of inliers

        if len(test_inliers) > len(best_inliers):
            best_model = A
            best_inliers = test_inliers

            # 5. Reevaluate number of iterations

            w = len(best_inliers) / len(pairs)
            if w == 1:  # <=> len(best_inliers) = len(pairs)
                break
            num_of_iterations = int(np.log(1 - p) / np.log(1 - w ** 4))
            print("Number of total RANSAC iterations:", num_of_iterations)
        cur_iteration += 1

    return best_model, best_inliers


def transform_to_base(A, im_low_res, im_high_res):
    """
    Given transformation matrix A which transforms from im_low_res to im_high_res,
    merge im_high_res onto im_low_res using A with backwards warping.
    :param A: Transformation matrix which transforms im_low_res to im_high_res coordinate system.
    :param im_low_res: Low resolution input image.
    :param im_high_res: High resolution input image.
    :return: Merged images after transformation.
    """

    mask = np.ones(im_low_res.shape)
    im_high_res_after_transform = np.zeros_like(im_low_res)

    for row in range(im_low_res.shape[0]):
        for col in range(im_low_res.shape[1]):

            # Find coordinate from first picture in second picture
            coor_orig = np.array([[row, col, 1]]).T
            coor_new_homogenous = np.matmul(A, coor_orig)
            coor_new = (coor_new_homogenous / coor_new_homogenous[2, 0])[:2, 0].flatten().astype(int)

            # Disregard negative coordinates.
            if np.any(coor_new < 0):
                continue
            try:
                # If alpha value of high-def image greater than 100, transform it.
                if im_high_res[coor_new[0], coor_new[1]][..., 3] > 100:

                    # Backward warping
                    im_high_res_after_transform[row, col] = im_high_res[coor_new[0], coor_new[1]][..., :3]
                    mask[row, col] = 0
                else:
                    # Ignore it otherwise.
                    im_high_res_after_transform[row, col] = 0
            except:
                pass

    # As the transformation isn't exactly on point, we need to make the transition between the two images smoother.
    # We do this by blurring the image with a Gaussian filter.
    mask = 1 - ndimage.gaussian_filter(1 - mask, 1)
    # return mask
    result = (1 - mask) * im_high_res_after_transform + mask * im_low_res

    return result.clip(0, 255).astype(np.uint8)


def transform_and_blend(im_low_res_path, im_high_res_path, threshold1, threshold2, ransac_epsilon):
    """
    Blend the two images after finding a transformation to align the images correctly.
    :param threshold1:
    :param ransac_epsilon:
    :param threshold2:
    :param im_low_res_path: image with low resolution.
    :param im_high_res_path: image with high resolution and an alpha channel.
    :return: the blended image
    """
    im_low_res = Image.open(im_low_res_path)
    im_high_res = Image.open(im_high_res_path)

    np_im_low_res = np.array(im_low_res)
    np_im_high_res = np.array(im_high_res)

    np_im_high_res_grayscale = np.array(im_high_res.convert("L"))
    np_im_low_res_grayscale = np.array(im_low_res.convert("L"))

    corners_high_res_coords = harris_corner_responses(np_im_high_res_grayscale, threshold1,
                                                      np.array(im_high_res)[..., 3])
    print("Calculated high resolution corner responses")

    corners_low_res_coords = harris_corner_responses(np_im_low_res_grayscale, threshold2)
    print("Calculated low resolution corner responses")
    print("Number of corners high res:", len(corners_high_res_coords), "Number of corners low res:",
          len(corners_low_res_coords))

    high_res_descriptors = get_descriptors(np_im_high_res_grayscale, corners_high_res_coords)
    print("Got high res mops descriptors")
    low_res_descriptors = get_descriptors(np_im_low_res_grayscale, corners_low_res_coords)
    print("Got low res mops descriptors")

    matching_pairs = find_matching_pairs(high_res_descriptors, low_res_descriptors)
    print("Number of pairs: " + str(len(matching_pairs)))

    A, inliers = ransac(matching_pairs, epsilon=ransac_epsilon)
    print("The transformation is: \n", A)
    print(str(len(inliers)) + " inliers.")

    return transform_to_base(A, np_im_low_res, np_im_high_res)



"""
# Example run:
from matplotlib import pyplot as plt

im_low_res_path = "img/lake_low_res.jpg"
im_high_res_path = "img/lake_high_res.png"
result = image_alignment.transform_and_blend(im_low_res_path, im_high_res_path, 1000, 1000, 5)
plt.imshow(result)
plt.show()
"""


