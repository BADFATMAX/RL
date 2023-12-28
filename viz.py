import cv2 as cv
import numpy as np
import os

BLACK = (0, 0, 0)
WIDTH = 1200
    
def Conv(canvas, n, i):
    ## Conv
    colors = [(83,49,0), (120,100,42), (160,158,95)]
    pos = WIDTH // n * (i-1)
    cv.rectangle(canvas, (pos+120, 40), (pos+40, 120), colors[0], -1)
    cv.rectangle(canvas, (pos+140, 60), (pos+60, 140), colors[1], -1)
    cv.rectangle(canvas, (pos+160, 80), (pos+80, 160), colors[2], -1) 
    return (pos, 140)
    ## Conv
    
def MaxPool(canvas, n, i):
    ## MaxPool
    colors = [(37,94,247), (20,117,255), (60,148,247)]
    pos = WIDTH // n * (i-1)
    cv.rectangle(canvas, (pos+100, 20), (pos+20, 100), colors[0], -1)
    cv.rectangle(canvas, (pos+140, 60), (pos+60, 140), colors[1], -1)
    cv.rectangle(canvas, (pos+180, 100), (pos+100, 180), colors[2], -1) 
    return (pos+15, 160)
    ## MaxPool

def BatchNorm(canvas, n, i):
    ## BatchNorm
    pos = WIDTH // n * (i-1)
    pt1 = (pos+115, 80)
    pt2 = (pos+85, 50)
    pt3 = (pos+85, 80)
    triangle_cnt = np.array( [pt1, pt2, pt3] )
    cv.drawContours(canvas, [triangle_cnt], 0, (1,190,229), -1)
    cv.rectangle(canvas, (pos+115, 80), (pos+85, 150), (1,190,229), -1)
    return (pos+15, 160)
    ## BatchNorm

def DropOut(canvas, n, i):
    ## DropOut
    pos = WIDTH // n * (i-1)
    p0, p1 = (pos+55, 55), (pos+145, 145)
    GRAY = (170,144,121)
    cv.rectangle(canvas, p0, p1, (215,189,166), -1)
    cv.line(canvas, (pos+85, 55), (pos+85, 145), GRAY, 5)
    cv.line(canvas, (pos+115, 55), (pos+115, 145), GRAY, 5)
    cv.line(canvas, (pos+55, 85), (pos+145, 85), GRAY, 5)
    cv.line(canvas, (pos+55, 115), (pos+145, 115), GRAY, 5)
    return (pos+15, 160)
    ## DropOut

def Flatten(canvas, n, i):
    ## Flattern
    pos = WIDTH // n * (i-1)
    cv.rectangle(canvas, (pos+90, 40), (pos+110, 160), (28,0,123), -1)
    cv.rectangle(canvas, (pos+75, 80), (pos+90, 120), (28,0,123), -1)
    ## Flatten

def Sigmoid(canvas, n, i, pos1, pos2):
    ## Sigmoid
    pos = WIDTH // n * (i-1)
    xs = np.linspace(-20, 20, 100)
    zs = 1/(1 + np.exp(-xs)) * 10
    xs += 100
    zs += 110
    cv.rectangle(canvas, (pos1+75+25, pos2), (pos1+125+25, pos2+50), (97,106,210), -1)
    for i in range(1, len(xs)):
        cv.line(canvas, (pos1 + int(xs[i-1])+25, pos2 - int(zs[i-1])+140), (pos1 +int(xs[i])+25, pos2 - int(zs[i])+140), BLACK, 5)
        # print(pos1 + int(xs[i]), pos2 - int(zs[i]))
    ## Sigmoid
        
def ReLU(canvas, n, i, pos1, pos2):
    ## RELU
    pos = WIDTH // n * (i-1)
    p0, p1, p2 = (pos1+80+25, pos2+35), (pos1+100+25, pos2+35), (pos1+120+25, pos2+15)
    GRAY = (135, 135, 135)
    cv.rectangle(canvas, (pos1+75+25, pos2), (pos1+125+25, pos2+50), GRAY, -1)
    cv.line(canvas, p0, p1, BLACK, 5)
    cv.line(canvas, p1, p2, BLACK, 5)
    ## RELU

def input_layer(canvas, n, i):
    ## Input
    pos = WIDTH // n * (i-1)
    cv.rectangle(canvas, (pos+75, 75), (pos+125, 125), (122,55,83), -1)
    ## Input

def output_layer(canvas, n, i):
    ## Output
    pos = WIDTH // n * (i-1)
    cv.rectangle(canvas, (pos+85, 10), (pos+115, 190), (122,55,83), -1)
    ## Output

Sequential = {
    "MaxPool2d": MaxPool,
    "Conv2d": Conv,
    "ReLU": ReLU,
    "Sigmoid": Sigmoid,
    "BatchNorm2d": BatchNorm,
    "Dropout2d": DropOut,
}

Activation = ["ReLU", "Sigmoid"]
    
def RenderToImage(fp):
    img = np.ones((200, WIDTH, 3), np.uint8) * 215
    net = ""
    if os.path.exists(fp):
        with open(fp, "r") as fptr:
            content = fptr.read()
            if not ("RENDER" in content):
                fptr.close()
                return img
            try:
                net = content.split("RENDER!")[-1].split("/RENDER")[0]
            except Exception:
                    return img
    else:
        return img

    if len(net) > 0:
        keySeq = list(Sequential.keys())
        test_list = net.split()
        res = []
        # x = []
        for j in test_list:
            for ll in keySeq:
                if j.find(ll) == 0:
                    test_list.remove(j)
                    res.append(ll)
        
        # printing result
        print("The filtered elements : " + str(res))
        actN = 0
        for j in Activation:
            actN += res.count(j)
        total = len(res) + 2 - actN
        print(total)

        input_layer(img, total, 1)
        idx = 2
        pos1, pos2 = None, None
        for el in res:
            print("cur: ", el)
            print("idx", idx)
            if el in Activation:
                (Sequential[f"{el}"])(img, total, idx, pos1, pos2)
            else:
                pos1, pos2 = (Sequential[f"{el}"])(img, total, idx)
                idx+=1
        output_layer(img, total, total)
        return img
    # cv.imshow("Net", img) 
    # cv.waitKey(0)
    # cv.destroyAllWindows()

# RenderToImage("./ex_nn_task/")