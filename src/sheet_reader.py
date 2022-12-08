import cv2
import matplotlib.pyplot as plt
import numpy as np

def dilate_img(img, kernel_size_x=3, kernel_size_y=3, iterations=1, show = False):
    kernel = np.ones((kernel_size_x, kernel_size_y), np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = iterations)
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(dilation)
        plt.show()
    return dilation

def erode_img(img, kernel_size_x=3, kernel_size_y=3, iterations=1, show = False):
    kernel = np.ones((kernel_size_x, kernel_size_y), np.uint8)
    erosion = cv2.erode(img,kernel,iterations = iterations)
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(erosion)
        plt.show()
    return erosion

def get_black_notes(img, show = False):
    dilated = dilate_img(img, 8, 6, show=show)
    eroded = erode_img(dilated, 8, 6, show=show)
    gray = np.float32(cv2.cvtColor(eroded,cv2.COLOR_BGR2GRAY))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel / sobel.max()*255).astype(np.uint8)
    black_white_threshold = 50
    sobel[sobel <= black_white_threshold] = 0
    sobel[sobel > black_white_threshold] = 255
    circles = cv2.HoughCircles(sobel, cv2.HOUGH_GRADIENT, dp = 1, minDist = 15, param1 = 1, param2 = 6, minRadius = 5, maxRadius = 7)
    circles = np.uint16(np.around(circles))
    new_circles = [] 
    img_circles = img.copy()
    for i in range(len(circles[0,:])):
        cv2.circle(img_circles, (circles[0,i,0], circles[0,i,1]), circles[0,i,2], (0,255,0), 2)
        new_circles.append(circles[0,i])   
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(img_circles)
        plt.show()
    return np.array([new_circles])

def get_white_notes(img, dp=1, minDist=20, param1=1000, param2=7, minRadius=6, maxRadius=7, show = False):
    gray = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    inverse = 255 - gray
    circles = cv2.HoughCircles(inverse, cv2.HOUGH_GRADIENT, dp = dp, minDist = minDist, param1 = param1, param2 = param2,
                                minRadius = minRadius, maxRadius = maxRadius)
    circles = np.uint16(np.around(circles))
    new_circles = []
    img_circles = img.copy()
    for i in range(len(circles[0,:])):
        if circles[0,i][0] > 70: # remove circles too close to the left
            clearest = max(img[circles[0,i][1]+x, circles[0,i][0]+j, 0] for x in range(-1,2) for j in range(-1,2))
            if clearest > 200: # if the center of the circle is not black
                cv2.circle(img_circles,(circles[0,i][0],circles[0,i][1]),circles[0,i][2],(0,255,0),2)
                new_circles.append(circles[0,i])
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(inverse, cmap='gray')
        plt.show()
        plt.figure(figsize=(16,6))
        plt.imshow(img_circles)
        plt.show()
    return np.array([new_circles])

def min_distance(center, circles):
    if len(circles) == 0:
        return 1000000
    return min(np.sqrt((center[0].astype(int) - circles[:,0].astype(int))**2 + (center[1].astype(int) - circles[:,1].astype(int))**2))

def all_notes_circles(img, black_circles, white_circles, min_dist=20, show=False):
    # we want the white_circle to be far enough from all the black circles
    new_white_circles = []
    for i in range(len(white_circles[0])):
        white_circle = white_circles[0,i]
        if min_distance(white_circle, black_circles[0]) > min_dist and min_distance(white_circle, white_circles[0,i+1:]) > min_dist:
            new_white_circles.append(white_circle)
    new_white_circles = np.array([new_white_circles])
    if show:
        img_circles = img.copy()
        for i in new_white_circles[0,:]:
            cv2.circle(img_circles,(i[0],i[1]),i[2],(0,0,255),2)
        for i in black_circles[0,:]:
            cv2.circle(img_circles,(i[0],i[1]),i[2],(0,255,0),2)
        plt.figure(figsize=(16,6))
        plt.imshow(img_circles)
        plt.show()
    all_circles = np.concatenate((black_circles, new_white_circles), axis=1)
    return all_circles, black_circles, new_white_circles

def isolate_index(index_lines_staff):
    group_staff = []
    for index in index_lines_staff:
        group = {index + 9*i for i in range(5)}
        if group.issubset(index_lines_staff):
            group_staff.append(sorted(list(group)))
    return group_staff

def group_circles_staff(circles, group_staff):
    circles_group_staff = [[] for _ in range(len(group_staff))]
    # there are 4 groups of lines of the staff
    # we want to associate each circle to a group of lines of the staff
    for i in range(len(circles[0])):
        circle = circles[0,i]
        dist = [np.abs(circle[1] - group[2]) for group in group_staff]
        circles_group_staff[np.argmin(dist)].append(circle)
    for i in range(len(circles_group_staff)):
        circles_group_staff[i] = sorted(circles_group_staff[i], key=lambda x: x[0])
    return circles_group_staff

def get_interline(lines):
    # Compute the average interline
    interline = round(np.mean([lines[i + 1] - lines[i] for i in range(len(lines) - 1)]))
    return interline

def get_base_notes_coordinates(lines):
    # Take the lines and the middles
    base_notes_coordinates = lines.copy()
    interline = get_interline(lines)
    base_notes_coordinates.append(lines[-1] + interline)
    base_notes_coordinates.append(lines[-1] + interline / 2)
    base_notes_coordinates.append(lines[0] - interline)
    base_notes_coordinates.append(lines[0] - interline / 2)
    for i in range(1, len(lines)):
        base_notes_coordinates.append(int((lines[i] + lines[i - 1]) / 2))
    base_notes_coordinates.sort()
    return base_notes_coordinates

def get_dic(base_notes_coordinates, clef_de_sol):
    # Associate the name according to the coordinates
    notes_clef_de_sol = ['do4', 'ré4', 'mi4', 'fa4', 'sol4', 'la4', 'si4', 'do5', 'ré5', 'mi5', 'fa5', 'sol5', 'la5'][::-1]
    notes_clef_de_fa = ['mi2', 'fa2', 'sol2', 'la2', 'si2', 'do3', 'ré3', 'mi3', 'fa3', 'sol3', 'la3', 'si3', 'do4'][::-1]
    dic_int_to_note = {}
    for i in range(len(notes_clef_de_sol)):
        if clef_de_sol:
            dic_int_to_note[base_notes_coordinates[i]] = notes_clef_de_sol[i]
        else:
            dic_int_to_note[base_notes_coordinates[i]] = notes_clef_de_fa[i]
    return dic_int_to_note

def get_notes_names(notes_coordinates, base_notes_coordinates, dic_int_to_note):
    # Attribute the closest note to each
    def closest_note(note):
        return base_notes_coordinates[np.array(abs(base_notes_coordinates - note[1])).argmin()]
    notes_names = []
    for note in notes_coordinates:
        notes_names.append(dic_int_to_note.get(closest_note(note)))
    return(notes_names)

def find_real_black_center(img, center, show=False):
    # Find the real center of the note by looking at the pixels around the center
    x, y = center
    # with a sliding window of size 10, we want the crop image to be as black as possible
    # we will use the squared sum of the pixels
    sum_pixels = np.sum(img[y-5:y+5, x-5:x+5])
    for a in range(x-10, x+10):
        for b in range(y-10, y+10):
            if a < 0 or b < 0 or a >= img.shape[1] or b >= img.shape[0]:
                continue
            sum_pixels_new = np.sum(img[b-5:b+5, a-5:a+5])
            if sum_pixels_new < sum_pixels:
                sum_pixels = sum_pixels_new
                x, y = a, b
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(img[center[1]-5:center[1]+5, center[0]-5:center[0]+5])
        plt.show()
        plt.figure(figsize=(16,6))
        plt.imshow(img[y-5:y+5, x-5:x+5])
        plt.show()
    return x, y

def find_real_white_center(nb_components, stats, centroids):
    # when two components are close, we want to get the center of the two components (ponderated by the size of the component)
    white_notes_coordinates = []
    for i in range(1, nb_components):
        for j in range(1, nb_components):
            if np.linalg.norm(centroids[i] - centroids[j]) < 20 and i != j:
                x = int((centroids[i][0] * stats[i][4] + centroids[j][0] * stats[j][4]) / (stats[i][4] + stats[j][4]))
                y = int((centroids[i][1] * stats[i][4] + centroids[j][1] * stats[j][4]) / (stats[i][4] + stats[j][4]))
                white_notes_coordinates.append((x, y))
                break
        else:
            white_notes_coordinates.append((int(centroids[i][0]), int(centroids[i][1])))
    return np.array([list(set(white_notes_coordinates))])

def find_small_components(img, show=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray < 180] = 0
    img_gray[img_gray >= 180] = 255
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_gray)
    for i in range(1, nb_components):
        # remove too big components and those that are too close to the left and right border
        if stats[i][4] > 40 or centroids[i][0] < 67 or centroids[i][0] > 940:
            output[output == i] = 0
    output[output > 0] = 255
    output = output.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(output)
    if show:
        plt.figure(figsize=(16,6))
        plt.imshow(output, cmap='jet')
        plt.show()
    return nb_components, output, stats, centroids

def pipeline_notes_staff(img, group_staff, img2=None, show=False):
    black_circles = get_black_notes(img, show = show)
    if img2 is None:
        white_circles = get_white_notes(img, dp=1, minDist=20, param1=1000, param2=7, minRadius=6, maxRadius=7, show = show)
    else:
        white_circles1 = get_white_notes(img, dp=1, minDist=20, param1=1000, param2=6, minRadius=6, maxRadius=7, show = show)
        white_circles2 = get_white_notes(img, dp=1, minDist=20, param1=1000, param2=7, minRadius=6, maxRadius=7, show = show)
        white_circles = np.concatenate((white_circles1, white_circles2), axis=1)
    _, black_circles, white_circles = all_notes_circles(img, black_circles, white_circles, min_dist=30, show=show)
    black_circles_center = []
    for circle in black_circles[0]:
        black_circles_center.append(find_real_black_center(img, (circle[0], circle[1])))
    black_circles_center = np.array([black_circles_center])
    group_black_circles_center = group_circles_staff(black_circles_center, group_staff)
    nb_components, output, stats, centroids = find_small_components(img, show)
    white_circles_center = find_real_white_center(nb_components, stats, centroids)
    group_white_circles_center = group_circles_staff(white_circles_center, group_staff)
    group_circles_center = []
    for i in range(len(group_staff)):
        group_circles_center.append(sorted(group_black_circles_center[i] + group_white_circles_center[i], key=lambda x: x[0]))
    return group_circles_center

def circles_to_notes_names(group_circles_center, group_staff):
    notes_names = []
    for i in range(len(group_staff)):
        dico = get_dic(get_base_notes_coordinates(group_staff[i]), (i % 2 == 0))
        notes_names.append(get_notes_names(group_circles_center[i], get_base_notes_coordinates(group_staff[i]), dico))
    return notes_names

def pipeline_number_mesure(img, show=False):
    dilated_img = dilate_img(img, kernel_size_x=50, kernel_size_y=1, iterations=1, show=show)
    x_mean1 =  np.mean(np.mean(dilated_img[:175,], axis=2), axis=0) # first two staff
    index_mesure1 = [x for x in np.where(x_mean1 < 200)[0] if 30 < x < 940]
    x_mean2 =  np.mean(np.mean(dilated_img[175:,], axis=2), axis=0) # third and fourth staff
    index_mesure2 = [x for x in np.where(x_mean2 < 200)[0] if 30 < x < 940]
    # sometimes, x and x+1 are both < 200, so we need to remove the duplicates 
    index_mesure1 = [x for x in index_mesure1 if x+1 not in index_mesure1]
    index_mesure2 = [x for x in index_mesure2 if x+1 not in index_mesure2]
    return [40]+index_mesure1+[940], [40]+index_mesure2+[940] # 30 and 940 are the left and right border of the image

def highlight_beat(img, mesure, beat, index_mesure, group_staff): # group_staff = [group_staff1, group_staff2]
    # we hightlight the zone between the two points of t
    new_img = img.copy()
    y_min, y_max = group_staff[0][0] - 20, group_staff[1][-1] + 20
    length_mesure = index_mesure[mesure] - index_mesure[mesure-1] - 10
    x_min = int(index_mesure[mesure-1] + length_mesure * (beat-1) / 3)
    x_max = int(x_min + length_mesure / 3)
    new_img[y_min:y_max, x_min:x_max, 0] = 0
    new_img[y_min:y_max, x_min:x_max, 2] = 0
    plt.figure(figsize=(16, 10))
    plt.imshow(new_img)
    plt.show()
    
def transform_to_beat(group_circles_center, index_mesure1, index_mesure2):
    group_beat = []
    for i in range(len(group_circles_center)):
        group_beat_i = []
        for j in range(len(group_circles_center[i])):
            mesures = index_mesure1 if i < 2 else index_mesure2
            for k in range(1, len(mesures)):
                if mesures[k-1] <= group_circles_center[i][j][0] < mesures[k]:
                    mesure = k
                    start_beat_2 = mesures[k-1]+(mesures[k]-mesures[k-1])/3
                    start_beat_3 = mesures[k-1]+2*(mesures[k]-mesures[k-1])/3
                    if mesures[k-1] <= group_circles_center[i][j][0] < start_beat_2:
                        beat = 1
                    elif start_beat_2 <= group_circles_center[i][j][0] < start_beat_3:
                        beat = 2
                    else:
                        beat = 3
                    group_beat_i.append([mesure, beat])
                    break
        group_beat.append(group_beat_i)
    return group_beat

def beat_to_infos(beat, index_mesures, group_staff, imgs, groups_beat, note_names):
    if beat <= (len(index_mesures[0])-1) * 3:
        beat_in_sheet = beat-1
        return (beat-1)%3+1, beat_in_sheet, index_mesures[0], group_staff[:2], imgs[0], groups_beat[0][:2], note_names[0][:2]
    elif beat <= sum([len(index_mesures[i])-1 for i in range(2)]) * 3:
        beat_in_sheet = beat-1-(len(index_mesures[0])-1)*3
        return (beat-1)%3+1, beat_in_sheet, index_mesures[1], group_staff[2:4], imgs[0], groups_beat[0][2:4], note_names[0][2:4]
    elif beat <= sum([len(index_mesures[i])-1 for i in range(3)]) * 3:
        beat_in_sheet = beat-1-sum([len(index_mesures[i])-1 for i in range(2)])*3
        return (beat-1)%3+1, beat_in_sheet, index_mesures[2], group_staff[:2], imgs[1], groups_beat[1][:2], note_names[1][:2]
    else:
        beat_in_sheet = beat-1-sum([len(index_mesures[i])-1 for i in range(3)])*3
        return (beat-1)%3+1, beat_in_sheet, index_mesures[3], group_staff[2:4], imgs[1], groups_beat[1][2:4], note_names[1][2:4]


def notes_in_beat(note_names, group_beat, mesure, beat):
    notes = [[],[]]
    for i in range(len(group_beat)):
        for j in range(len(group_beat[i])):
            if group_beat[i][j][0] == mesure and group_beat[i][j][1] == beat:
                notes[1-i].append(note_names[i][j])
    return notes