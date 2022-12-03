import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_hand_region(frame, threshold=30, show=False):
    new_frame = frame.std(axis=2)
    new_frame[:30,:] = 0
    new_frame[new_frame < threshold] = 0
    new_frame[new_frame >= threshold] = 255
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(new_frame, cmap='gray')
        plt.show()
    return new_frame

def get_list_pixels(image_hand, key_number, labels, above = False): # above > 0 if we want the pixels to be far from fingers
    # We check if the key k is pressed by checking all pixels in the key k that are not in the hands
    result = np.where((labels == key_number) & (image_hand[:,:,0] < 10))
    if not above:
        return result
    pixels_hided = np.where((labels == key_number) & (image_hand[:,:,0] > 200))
    # pixels around the hand shouldn't be considered because the color is influenced
    if len(pixels_hided[0]) > 0:
        highest_pixel = pixels_hided[0].min()
    else :
        highest_pixel = 120 # in fact, we shouldn't consider this because no finger above the key
    return (result[0][(result[0] > 10) & (result[0] <= min(110,max(0,highest_pixel-above)))],
            result[1][(result[0] > 10) & (result[0] <= min(110,max(0,highest_pixel-above)))])

def get_list_pixels_faster_computation(image_hand, key_number, labels, centroids, above = False):
    # above > 0 if we want the pixels to be far from fingers
    x,y = centroids[key_number]
    # we want to check only the pixels around the centroid, say x-20 to x+20 to make the computation faster
    region_to_check_labels = labels[:, int(x-20):int(x+20)]
    region_to_check_hand = image_hand[:, int(x-20):int(x+20)]
    result = np.where((region_to_check_labels == key_number) & (region_to_check_hand[:,:,0] < 10))
    result = (result[0], result[1]+int(x-20))
    if not above:
        return result
    pixels_hided = np.where((region_to_check_labels == key_number) & (region_to_check_hand[:,:,0] > 200))
    pixels_hided = (pixels_hided[0], pixels_hided[1]+int(x-20))
    # pixels around the hand shouldn't be considered because the color is influenced
    if len(pixels_hided[0]) > 0:
        highest_pixel = pixels_hided[0].min()
    else :
        highest_pixel = 120
    return (result[0][(result[0] > 10) & (result[0] <= min(110,max(0,highest_pixel-above)))],
            result[1][(result[0] > 10) & (result[0] <= min(110,max(0,highest_pixel-above)))])

def average_color(image, list_pixels):
    return np.mean(image[list_pixels[0], list_pixels[1]], axis=0)

def highlight_keys(image_hand, image_diff, threshold, labels, centroids):
    new_image = image_hand.copy()
    for i in range(1,89):
        list_testing = get_list_pixels_faster_computation(image_hand, i, labels, centroids, above=10)
        color = np.mean(image_diff[list_testing[0], list_testing[1]], axis=0)
        mean_color = np.mean(color)
        if mean_color > threshold:
            new_image[np.where(labels == i)] = [0,mean_color,0]
        elif mean_color > 50:
            new_image[np.where(labels == i)] = [mean_color,0,0]
    return new_image

def pipeline_key_pressed(img, mask, labels, centroids, threshold_hand=30, above=10, threshold_confidence=150, show=False):
    image1 = get_hand_region(img, threshold_hand)
    image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2RGB)
    image2 = img - mask + 10
    image = highlight_keys(image1, image2, threshold_confidence, labels, centroids)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(image)
        plt.show()
    return image