import cv2
import numpy as np
import os

imput_file_dir = "C:\\Users\\Administrator\\Desktop\\EP_mean\\black_background_dataset\\"
output_file_dir = "C:\\Users\\Administrator\\Desktop\\EP_mean\\"
mean_total = []
ST_total = []

lst = os.listdir(imput_file_dir)
for i in lst:
    input_file_path = os.path.join(imput_file_dir, i)
    img = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
    print(cv2.meanStdDev(img))
    mean, ST = cv2.meanStdDev(img)
    mean_squeeze = np.squeeze(mean)
    ST_squeeze = np.squeeze(ST)
    print(mean_squeeze)
    print("##########")
    print(ST_squeeze)
    mean_total.append(mean_squeeze)
    ST_total.append(ST_squeeze)

output_file_path_mean = os.path.join(output_file_dir, "mean_gray.txt")
output_file_path_ST = os.path.join(output_file_dir, "ST_gray.txt")

with open(output_file_path_mean, 'w') as f:
    f.write("EP_mean_GRAY")
    f.write(",")
    f.write("image_name")
    f.write("\n")
    for i in range(len(mean_total)):
        f.write(str(format(mean_total[i], '.4f')))
        f.write(",")
        f.write(lst[i])
        f.write("\n")

with open(output_file_path_ST, 'w') as f:
    f.write("EP_ST_GRAY")
    f.write(",")
    f.write("image_name")
    f.write("\n")
    for i in range(len(ST_total)):
        f.write(str(format(ST_total[i], '.4f')))
        f.write(",")
        f.write(lst[i])
        f.write("\n")

