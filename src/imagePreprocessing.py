import numpy as np
import cv2


def binarize(img):
    # binaries the image by illuminating + thresholding(OTSU)

    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    ret, th = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return th


def get_contours(img):
    # Get contours but first invert the colors for efficiency
    
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_wordCoordinate_list(contours, height_of_image, width_of_image):
    # This function returns the extreme coordinates of each contour and average character height

    min_x, min_y = width_of_image, height_of_image
    max_x = max_y = 0
    avgheight = 0
    total = 0
    imglt = list()
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 10 and h > 10:
            avgheight += h
            total += 1
            v= (x, x+w, y, y+h)
            imglt.append(v)

    avgheight /= total
    return (imglt, avgheight)


def sort_wordCoordinate_list(imglt, avgheight):
    # Returns sorted contour list (list of list of contour_tuple)

    newlt = sorted(imglt, key=lambda x:x[2])
    a, b, c, d = newlt[0]
    mainlt = list()
    lt = list()
    s = (7*avgheight)/10
    for elm in newlt:
        c1 = elm[2]
        if c1 - c > s:
            mainlt.append(lt)
            lt = list()
            
        c = c1
        lt.append(elm)
    mainlt.append(lt)

    for elm in mainlt:
        elm.sort(key=lambda x:x[0])

    return mainlt


def join_contours_of_same_words(mainlt, avgheight):
    # Returns combined frame coordinate list

    finallt = list()
    
    for elmm in mainlt:
        flag = 0
        for i in range(len(elmm)-1):
            flag = 0
            a, b, c, d = elmm[i]
            p, q, r, s = elmm[i+1]
            fvalue = elmm[i+1]
            if (p-b) < (3*avgheight)/5:
                xmin, xmax, ymin, ymax = min(a,p), max(b,q), min(c,r), max(d,s)
                v = (xmin, xmax, ymin, ymax)
                try:
                    finallt.remove(elmm[i])
                except:
                    pass
                finallt.append(v)
                elmm[i+1] = v
                elmm[i] = v
                flag = 1

            else:
                v = elmm[i]
                finallt.append(v)

        if flag == 0:
            finallt.append(fvalue)

    return finallt


def get_word_list(img):
    # Driver function to preprocess the image

    thresholded_img = binarize(img)
    contours = get_contours(thresholded_img)
    word_list, avgheight = get_wordCoordinate_list(contours, img.shape[0], img.shape[1])
    main_list = sort_wordCoordinate_list(word_list, avgheight)
    final_list = join_contours_of_same_words(main_list, avgheight)

    return final_list
    

def preProcess(im):

    img = cv2.imread(im, 0)
    # frame = img.copy()
    finallt = get_word_list(img)
    # cv2.waitKey()
    return finallt


if __name__ == "__main__":

    img = cv2.imread('../data/test_segmentation.png', 0)
    frame = img.copy()
    finallt = get_word_list(img)
    for v in finallt:
        a, b, c, d = v
        cv2.rectangle(frame, (a,c), (b,d), (0, 0, 255), 3)

    cv2.imwrite("../sample/output_segmentation.png", frame)
    cv2.imshow("This should show the words enclosed in boxes", cv2.resize(frame, (960, 540)))
    cv2.waitKey()
