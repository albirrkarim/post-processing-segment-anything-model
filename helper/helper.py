# Tools
import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

import sys
import cv2


def getFileName(img_path, id=None):
    if (id == None):
        file_name = os.path.splitext(os.path.basename(img_path))[0]
    else:
        file_name = id

    return file_name


def cropUnusedBlankPixel(myImage):
    # https://stackoverflow.com/a/53829086
    if isinstance(myImage, Image.Image):
        PIL_image = myImage
    else:
        PIL_image = Image.fromarray(myImage.astype('uint8'), 'RGBA')

    bounding = PIL_image.getbbox()

    if (bounding != None):
        PIL_image = PIL_image.crop(bounding)
    else:
        width, height = PIL_image.size
        bounding = (0, 0, width, height)

    return [PIL_image, np.array(bounding).astype(int)]


def newMaskApply(image, mask):
    s1 = image.shape  # h,w,c
    s2 = mask.shape  # h,w
    if (s1[0] == s2[0] and s1[1] == s2[1]):
        image[mask == 0] = [0, 0, 0, 0]

    return image


def extendBoundaries(img, bounding, pixelToExtend=30):
    x = bounding[0]-pixelToExtend
    y = bounding[1]-pixelToExtend
    x1 = bounding[2]+pixelToExtend
    y1 = bounding[3]+pixelToExtend

    if (x <= 0):
        x = bounding[0]

    if (y <= 0):
        y = bounding[1]

    w = img.shape[1]
    h = img.shape[0]

    if (y1 > h):
        y1 = h

    if (x1 > w):
        x1 = w

    # if(y1 <= y):
    #     y = bounding[1]
    #     y1 = bounding[3]

    # if(x1 <= x):
    #     x = bounding[0]
    #     x1 = bounding[2]

    return [x, y, x1, y1]


def cropToBoundingBox(img, bounding):
    x = bounding[0]
    y = bounding[1]
    x1 = bounding[2]
    y1 = bounding[3]

    result = img[y:y1, x:x1]
    return result


def patchToBoundingBox(img, bounding, imgPatch):
    # Give the transparent image, the segment image
    x = bounding[0]
    y = bounding[1]
    x1 = bounding[2]
    y1 = bounding[3]

    img[y:y1, x:x1] = imgPatch
    return img


def enhanceEdgeMod(targetImage, imgOrigin, threshold=50):
    # Enhance with original image
    maskEnhance = targetImage[:, :, 3] > threshold

    targetImage[maskEnhance] = imgOrigin[maskEnhance]
    return targetImage


def blurLevel(np_img):
    # Test the precentage

    mask = (np_img[:, :, 3] > 0) & (np_img[:, :, 3] < 200)
    m = np_img[mask]

    blur_region = m.shape[0]
    all_square = np_img.shape[0] * np_img.shape[1]
    blur_level = blur_region/all_square

    return blur_level


def useRGBImage(imgPathSource):
    img = Image.open(imgPathSource)

    if (img.format == 'PNG'):
        file_name = os.path.splitext(os.path.basename(imgPathSource))[0]
        sourceDir = os.path.dirname(imgPathSource)
        expect_img_path = sourceDir + "/" + file_name + "_rgb.jpg"

        PIL_image = Image.open(imgPathSource).convert("RGB")
        PIL_image.save(expect_img_path)
        return expect_img_path

    return imgPathSource


def split_objects(PIL_Image, last_index, output_folder, file_name):

    # Chat GPT
    # imagine you have an rgba image, inside it have two or more objects that separated with transparent area.
    # split all object based on the transparent area that sparated them.
    # then save that objects into different image.
    # only save the image if the object area is more than 1000 pixel square

    index_save = 1
    outputs = []
    # Load the image
    image = PIL_Image
    pixels = image.load()
    width, height = image.size

    # Keep track of which pixels we've already processed
    processed = set()

    # Iterate over all the pixels in the image
    for x in range(width):
        for y in range(height):
            # If this pixel has already been processed or it's transparent, skip it
            if (x, y) in processed or pixels[x, y][3] == 0:
                continue

            # Use a flood fill algorithm to find all the pixels that belong to this object
            object_pixels = set()
            stack = [(x, y)]
            while stack:
                px, py = stack.pop()
                if px < 0 or px >= width or py < 0 or py >= height:
                    continue
                if (px, py) in processed or pixels[px, py][3] == 0:
                    continue
                object_pixels.add((px, py))
                processed.add((px, py))
                stack.append((px-1, py))
                stack.append((px+1, py))
                stack.append((px, py-1))
                stack.append((px, py+1))

            # If the object has an area less than 1000 pixels, skip it
            if len(object_pixels) < 1000:
                continue

            # Create a new image with just the pixels for this object
            object_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            object_pixels_list = list(object_pixels)
            for px, py in object_pixels_list:
                object_image.putpixel((px, py), pixels[px, py])

            # Save the object image to disk with a unique name
            nextIndex = last_index+index_save
            new_file_name = f"{file_name}_dom_segs_{nextIndex}.png"
            index_save += 1

            PIL_image, bounding = cropUnusedBlankPixel(object_image)

            PIL_image.save(output_folder+new_file_name)

            # Save for the output
            outputs.append([nextIndex, new_file_name, bounding])

    return outputs


def display_images_in_column(images, titles):
    # Calculate the number of rows and columns
    num_images = len(images)
    num_columns = 2
    num_rows = int(np.ceil(num_images / num_columns))

    # Display the images using matplotlib.pyplot
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 7 * num_rows))

    for idx, img in enumerate(images):
        row, col = divmod(idx, num_columns)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(titles[idx])
        axes[row, col].axis('off')

    fig.subplots_adjust(hspace=0, wspace=0.1)

    # Remove unused subplots
    for idx in range(num_images, num_rows * num_columns):
        row, col = divmod(idx, num_columns)
        fig.delaxes(axes[row, col])

    plt.show()


def canny_edge_detection(image_path, lower_threshold=50, upper_threshold=100):
    # Load the image

    image = None

    if isinstance(image_path, str):
        # print("string_var is a string")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        if isinstance(image_path, np.ndarray):
            # print("numpy_var is a NumPy array")
            # print("numpy_var is not a NumPy array")
            image = cv2.cvtColor(image_path, cv2.IMREAD_GRAYSCALE)

        # print("string_var is not a string")

    if image is None:
        print("Error: Unable to read the image.")
        sys.exit(1)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

    return edges


def sobel_edge_detection(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to read the image.")
        sys.exit(1)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Calculate the gradients using Sobel operators
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the absolute gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)

    # Combine the gradients
    combined_sobel = np.sqrt(np.square(abs_sobel_x) + np.square(abs_sobel_y))

    # Scale and convert the gradients to an 8-bit image
    scaled_sobel = np.uint8(255 * combined_sobel / np.max(combined_sobel))

    return scaled_sobel
