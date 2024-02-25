import os
import cv2
import numpy as np
import datetime

def extract_green_regions(image_path, output_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 100, 100])  # Lower threshold for green color in HSV
    upper_green = np.array([75, 255, 255])  # Upper threshold for green color in HSV

    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img, img, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Filter out small regions
            x, y, w, h = cv2.boundingRect(contour)
            green_region = img[y:y+h, x:x+w]
            green_regions.append(green_region)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imwrite(output_path, img)
    return green_regions


if __name__ == "__main__":
    image_folder = 'maskrcnn-test/generated/'
    output_folder = 'maskrcnn-test/generated/'
    os.makedirs(output_folder, exist_ok=True)

    image_files = [file for file in os.listdir(image_folder) if file.endswith('.jpg')]

    # start_time = datetime.datetime.now()
    for i, image_file in enumerate(image_files):
        print("1:", i)
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, f'green_{i}.jpg')
        green_regions = extract_green_regions(image_path, output_path)
    
        for j, region in enumerate(green_regions):
            print("2:", j)
            region_output_path = os.path.join(output_folder, f'region_{i}.png')
            cv2.imwrite(region_output_path, region)

    # total_time = datetime.datetime.now() - start_time
    # print('Extraction time: {}'.format(total_time))
