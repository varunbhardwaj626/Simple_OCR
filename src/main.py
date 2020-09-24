from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import io
import editdistance
import numpy as np
import random
import PySimpleGUI as sg
from Model import Model, DecoderType
from imagePreprocessing import preProcess


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/test.png'
    fnCorpus = '../data/corpus.txt'


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


def preprocess(img, imgSize, dataAugmentation=False):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    if dataAugmentation:
        stretch = (random.random() - 0.5)
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
        img = cv2.resize(img, (wStretched, img.shape[0]))

    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    img = cv2.transpose(target)

    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img


def infer(model, fnImg, arr):
    "recognize text in image provided by file path"
    s = ''
    p = '~'
    im = cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE)
    for i in arr:
        a, b, c, d = i
        if b-a < 5 or d-c < 5 :
            continue
        img = preprocess(im[c:d, a:b], Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, True)
        v = recognized[0]
        if v != p:
            s += v + ' '
            p = v
    print(s)
    return s


def mainfn(filename, arr):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(),
                  decoderType, mustRestore=True, dump=args.dump)
    return infer(model, filename, arr)


sg.theme('Dark Brown')
filename = 'default.png'
layout = [
    [sg.Input(key='file'), sg.FileBrowse(key='browse')],
    [sg.Button('Read Image')],
    [sg.Image(filename, key='image', size=(300, 300))],
    [sg.Text('Founded Text : ')], [sg.Multiline(size=(50, 4),
                                                text_color='black', key="output", background_color='white')],

    [sg.Button('Exit')]]

window = sg.Window('Handwriting Detector', layout)

while True:
    event, values = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    if event == 'Browse':
        window['file'].update(values['browse'])
    if event == 'Read Image':
        filename = values['browse']

        try:
            im=cv2.imread(filename)
            im=cv2.resize(im, (300, 300)) 
            cv2.imwrite('intemediate.png',im)
            window['image'].Update('intemediate.png')
          
        except:
            print('CAN\'T PUT THE IMAGE ON CANVAS')
        arr = preProcess(filename)
        image_to_text = mainfn(filename, arr)
        window['output'].update(image_to_text)
window.close()
