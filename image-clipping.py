import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob as gb

# Extracting corner points using edge detection
def corner_extraction(figure):
    edge = cv2.Canny(figure, 300, 1000)  #Edge detection using Canny algorithm
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #Contour extraction
    cnt = contours[0]
    print(cnt)
    peri = cv2.arcLength(cnt, True)                              #Calculate the contour perimeter
    print(peri)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)              #Polygon approximation to extract polygon corner points
    i = 1
    while approx.size < 8 or approx[0, 0, 0] > 55:               #Determine whether the current contour is a quadrilateral contour or a part of it
        cnts = contours[i]                                       #If it is not part of the quadrilateral contour, select the next contour
        test = edge.copy()
        cv2.drawContours(test, contours, i, (0, 255, 0), 5)
        cv2.imshow('test', test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        peri = cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, 0.02*peri, True)
        i += 1
        print(i)
    return approx

# Contour extraction (for image cropping)
def contour_extraction(figure_1):
    edge = cv2.Canny(figure_1, 200, 250)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    return cnt

# Calculation of the position of the target graphic (coordinates of the reference point)
def located_point(corner_point, center_point, lattice):
    center_point_array_origin = np.array(center_point)
    size = len(center_point)
    center_point_array = np.zeros((int(size), 2), dtype=float)              #Reshape the matrix
    for i in range(0, int(size)):                                           #Find the actual coordinates of the actual reference point
        center_point_array[i, 0] = corner_point[0, 0] + lattice / 2 * center_point_array_origin[i, 0] + center_point_array_origin[i, 1] * lattice - lattice / 2
        center_point_array[i, 1] = corner_point[0, 1] + lattice / 2 * math.sqrt(3) * center_point_array_origin[i, 0] - lattice / (2 * math.sqrt(3))
    return center_point_array

# The contour coordinate points of the inverted triangle target figure
def invert_triangle_contour_points(center_point, lattice):
    contour_points_array = np.zeros((12,2), dtype=float)
    #A
    contour_points_array[0, 0] = center_point[0, 0] - lattice
    contour_points_array[0, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #B
    contour_points_array[1, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[1, 1] = center_point[0, 1] - lattice * 3 / (math.sqrt(3) * 2)
    #C
    contour_points_array[2, 0] = center_point[0, 0]
    contour_points_array[2, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #D
    contour_points_array[3, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[3, 1] = center_point[0, 1] - lattice * 3 / (math.sqrt(3) * 2)
    #E
    contour_points_array[4, 0] = center_point[0, 0] + lattice
    contour_points_array[4, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #F
    contour_points_array[5, 0] = center_point[0, 0] + lattice
    contour_points_array[5, 1] = center_point[0, 1]
    #G
    contour_points_array[6, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[6, 1] = center_point[0, 1] + lattice / (math.sqrt(3) * 2)
    #H
    contour_points_array[7, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[7, 1] = center_point[0, 1] + lattice * 3 / (math.sqrt(3) * 2)
    #I
    contour_points_array[8, 0] = center_point[0, 0]
    contour_points_array[8, 1] = center_point[0, 1] + lattice * 2 / math.sqrt(3)
    #J
    contour_points_array[9, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[9, 1] = center_point[0, 1] + lattice * 3 / (math.sqrt(3) * 2)
    #K
    contour_points_array[10, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[10, 1] = center_point[0, 1] + lattice / (math.sqrt(3) * 2)
    #L
    contour_points_array[11, 0] = center_point[0, 0] - lattice
    contour_points_array[11, 1] = center_point[0, 1]
    return contour_points_array

# Outline coordinates of the target shape of an equilateral triangle
def triangle_contour_points(center_point, lattice):
    contour_points_array = np.zeros((12, 2), dtype=float)
    #A
    contour_points_array[0, 0] = center_point[0, 0]
    contour_points_array[0, 1] = center_point[0, 1]
    #B
    contour_points_array[1, 0] = center_point[0, 0]
    contour_points_array[1, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #C
    contour_points_array[2, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[2, 1] = center_point[0, 1] - 3 * lattice / (2 * math.sqrt(3))
    #D
    contour_points_array[3, 0] = center_point[0, 0] + lattice
    contour_points_array[3, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #E
    contour_points_array[4, 0] = center_point[0, 0] + lattice
    contour_points_array[4, 1] = center_point[0, 1]
    #F
    contour_points_array[5, 0] = center_point[0, 0] + 3 * lattice / 2
    contour_points_array[5, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    #G
    contour_points_array[6, 0] = center_point[0, 0] + 3 * lattice /2
    contour_points_array[6, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    #H
    contour_points_array[7, 0] = center_point[0, 0] + lattice
    contour_points_array[7, 1] = center_point[0, 1] + 2 * lattice / math.sqrt(3)
    #I
    contour_points_array[8, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[8, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    #J
    contour_points_array[9, 0] = center_point[0, 0]
    contour_points_array[9, 1] = center_point[0, 1] + 2 * lattice / math.sqrt(3)
    #K
    contour_points_array[10, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[10, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    #L
    contour_points_array[11, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[11, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    return contour_points_array

# Contour coordinates of double hexagon type 1 target graphics
def double_hexagon_1_points(center_point, lattice):
    contour_points_array = np.zeros((10, 2), dtype=float)
    #A
    contour_points_array[0, 0] = center_point[0, 0]
    contour_points_array[0, 1] = center_point[0, 1]
    #B
    contour_points_array[1, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[1, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    #C
    contour_points_array[2, 0] = center_point[0, 0] - lattice
    contour_points_array[2, 1] = center_point[0, 1]
    #D
    contour_points_array[3, 0] = center_point[0, 0] - lattice
    contour_points_array[3, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #E
    contour_points_array[4, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[4, 1] = center_point[0, 1] - 3 * lattice / (2 * math.sqrt(3))
    #F
    contour_points_array[5, 0] = center_point[0, 0]
    contour_points_array[5, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #G
    contour_points_array[6, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[6, 1] = center_point[0, 1] - 3 * lattice / (2 * math.sqrt(3))
    #H
    contour_points_array[7, 0] = center_point[0, 0] + lattice
    contour_points_array[7, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    #I
    contour_points_array[8, 0] = center_point[0, 0] + lattice
    contour_points_array[8, 1] = center_point[0, 1]
    #J
    contour_points_array[9, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[9, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    return contour_points_array

# Contour coordinates of double hexagon type 2 target graphics
def double_hexagon_2_points(center_point, lattice):
    contour_points_array = np.zeros((10, 2), dtype=float)
    # A
    contour_points_array[0, 0] = center_point[0, 0]
    contour_points_array[0, 1] = center_point[0, 1]
    # B
    contour_points_array[1, 0] = center_point[0, 0]
    contour_points_array[1, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    # C
    contour_points_array[2, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[2, 1] = center_point[0, 1] - 3 * lattice / (2 * math.sqrt(3))
    # D
    contour_points_array[3, 0] = center_point[0, 0] + lattice
    contour_points_array[3, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    # E
    contour_points_array[4, 0] = center_point[0, 0] + lattice
    contour_points_array[4, 1] = center_point[0, 1]
    # F
    contour_points_array[5, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[5, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    # G
    contour_points_array[6, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[6, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    # H
    contour_points_array[7, 0] = center_point[0, 0]
    contour_points_array[7, 1] = center_point[0, 1] + 2 * lattice / math.sqrt(3)
    # I
    contour_points_array[8, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[8, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    # J
    contour_points_array[9, 0] = center_point[0, 0] - lattice / 2
    contour_points_array[9, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    return contour_points_array

# Contour coordinates of double hexagon type 3 target graphics
def double_hexagon_3_points(center_point, lattice):
    contour_points_array = np.zeros((10, 2), dtype=float)
    # A
    contour_points_array[0, 0] = center_point[0, 0]
    contour_points_array[0, 1] = center_point[0, 1]
    # B
    contour_points_array[1, 0] = center_point[0, 0]
    contour_points_array[1, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    # C
    contour_points_array[2, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[2, 1] = center_point[0, 1] - 3 * lattice / (2 * math.sqrt(3))
    # D
    contour_points_array[3, 0] = center_point[0, 0] + lattice
    contour_points_array[3, 1] = center_point[0, 1] - lattice / math.sqrt(3)
    # E
    contour_points_array[4, 0] = center_point[0, 0] + lattice
    contour_points_array[4, 1] = center_point[0, 1]
    # F
    contour_points_array[5, 0] = center_point[0, 0] + 3 * lattice / 2
    contour_points_array[5, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    # G
    contour_points_array[6, 0] = center_point[0, 0] + 3 * lattice / 2
    contour_points_array[6, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    # H
    contour_points_array[7, 0] = center_point[0, 0] + lattice
    contour_points_array[7, 1] = center_point[0, 1] + 2 * lattice / math.sqrt(3)
    # I
    contour_points_array[8, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[8, 1] = center_point[0, 1] + 3 * lattice / (2 * math.sqrt(3))
    # J
    contour_points_array[9, 0] = center_point[0, 0] + lattice / 2
    contour_points_array[9, 1] = center_point[0, 1] + lattice / (2 * math.sqrt(3))
    return contour_points_array

# Traverse all images in a folder
img_EP_1_path = gb.glob("G:\computational data\electrostatic-potential database\EP-2\\*.jpg")

# The relative coordinates of the reference point of the target graphic
center_point = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]

save_path = 'G:\computational data\electrostatic-potential database\Database-ep-2\\'  # 保存路径
count = 0
count_0 = 1
for path in img_EP_1_path:
    origin = cv2.imread(path, cv2.IMREAD_COLOR)
    points = corner_extraction(origin)
    n = 0
    while abs(points[n, 0, 1] - points[1, 0, 1]) > 200 or abs(points[n, 0, 0] - points[1, 0, 0]) < 200: #Find the two endpoints of the bottom edge to calculate the lattice constant (the conditions for finding the bottom edge endpoints: the difference between the x coordinates is greater than 200, and the difference between the y coordinates is less than 200)
        print(abs(points[n, 0, 1] - points[1, 0, 1]))
        print(abs(points[n, 0, 0] - points[1, 0, 0]))
        n += 1
    lattice_constant = abs(points[n, 0, 0] - points[1, 0, 0]) / 6
    size = points.size
    points_reshape = points.reshape((int(size/2), 2))

    #Inverted triangle target
    IT_center_points = located_point(points_reshape, center_point, lattice_constant)
    for i in range(0, 9):
        mask = origin.copy()  # Construct a clipping mask the same size as the original image
        mask.fill(255)  # Mask Whitening
        center_one_point = IT_center_points[i:i + 1]  # Calculation of actual coordinates of reference points
        contour_point = invert_triangle_contour_points(center_one_point, lattice_constant)  # Calculation of contour points of target graphics
        print(contour_point)
        contour_points_reshape = contour_point.reshape((-1, 1, 2))  # Changing the outline point matrix shape to facilitate polygon fill input
        print(contour_points_reshape)
        cv2.fillPoly(mask, np.int32([contour_points_reshape]), (0, 0, 0))  # The target polygon is filled with black
        cut_figure = cv2.add(origin, mask)  # Image addition to achieve mask clipping
        cut_contour = contour_extraction(cut_figure)  # Extract the contour of the cropped image
        x, y, w, h = cv2.boundingRect(cut_contour)  # Extraction of rectangle boundary coordinates and size
        output = cut_figure[y:y + h, x:x + w]  # Image Cropping
        count += 1
        cv2.imwrite(save_path + str(count) + '.jpg', output)  # Image Output

    # Triangle target 
    T_center_points = located_point(points_reshape, center_point, lattice_constant)
    for i in range(0, 9):
        mask = origin.copy()
        mask.fill(255)
        center_one_point = T_center_points[i:i + 1]
        contour_point = triangle_contour_points(center_one_point, lattice_constant)
        contour_points_reshape = contour_point.reshape((-1, 1, 2))
        print(contour_points_reshape)
        cv2.fillPoly(mask, np.int32([contour_points_reshape]), (0, 0, 0))
        cut_figure = cv2.add(origin, mask)
        cut_contour = contour_extraction(cut_figure)
        x, y, w, h = cv2.boundingRect(cut_contour)
        output = cut_figure[y:y + h, x:x + w]
        count += 1
        cv2.imwrite(save_path + str(count) + '.jpg', output)

    # Double Hexagon Type 1
    DH1_center_points = located_point(points_reshape, center_point, lattice_constant)
    for i in range(0, 9):
        mask = origin.copy()
        mask.fill(255)
        center_one_point = DH1_center_points[i:i + 1]
        contour_point = double_hexagon_1_points(center_one_point, lattice_constant)
        contour_points_reshape = contour_point.reshape((-1, 1, 2))
        cv2.fillPoly(mask, np.int32([contour_points_reshape]), (0, 0, 0))
        cut_figure = cv2.add(origin, mask)
        cut_contour = contour_extraction(cut_figure)
        x, y, w, h = cv2.boundingRect(cut_contour)
        output = cut_figure[y:y + h, x:x + w]
        count += 1
        cv2.imwrite(save_path + str(count) + '.jpg', output)

    # Double Hexagon Type 2
    DH2_center_points = located_point(points_reshape, center_point, lattice_constant)
    for i in range(0, 9):
        mask = origin.copy()
        mask.fill(255)
        center_one_point = DH2_center_points[i:i + 1]
        contour_point = double_hexagon_2_points(center_one_point, lattice_constant)
        contour_points_reshape = contour_point.reshape((-1, 1, 2))
        cv2.fillPoly(mask, np.int32([contour_points_reshape]), (0, 0, 0))
        cut_figure = cv2.add(origin, mask)
        cut_contour = contour_extraction(cut_figure)
        x, y, w, h = cv2.boundingRect(cut_contour)
        output = cut_figure[y:y + h, x:x + w]
        count += 1
        cv2.imwrite(save_path + str(count) + '.jpg', output)

    # Double Hexagon Type 3
    DH3_center_points = located_point(points_reshape, center_point, lattice_constant)
    for i in range(0, 9):
        mask = origin.copy()
        mask.fill(255)
        center_one_point = DH3_center_points[i:i + 1]
        contour_point = double_hexagon_3_points(center_one_point, lattice_constant)
        contour_points_reshape = contour_point.reshape((-1, 1, 2))
        cv2.fillPoly(mask, np.int32([contour_points_reshape]), (0, 0, 0))
        cut_figure = cv2.add(origin, mask)
        cut_contour = contour_extraction(cut_figure)
        x, y, w, h = cv2.boundingRect(cut_contour)
        output = cut_figure[y:y + h, x:x + w]
        count += 1
        cv2.imwrite(save_path + str(count) + '.jpg', output)
    count_0 += 1