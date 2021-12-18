# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:44:41 2021

@author: dantr
"""
#Libraries used
# https://pypi.org/project/svgpathtools/
# https://github.com/3b1b/manim

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import animation


pointArrayX = []
pointArrayY = []
CornerArray = []

class Corner():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def readPic(filename):
    temp = []
    img = cv2.imread(filename, 0)
    for (x, y) in cv2.goodFeaturesToTrack(img, 0, 0.01, img.shape[0] / 20)[:, 0].astype("int0"):
        cv2.circle(img, (x, y), 27, 127, -1)
        temp.append(Corner(x,y))
    temp.sort(key=lambda corner: (corner.x, corner.y))
    BL = temp[0]
    temp.sort(key=lambda corner: (corner.x, -corner.y))
    TL = temp[0]
    temp.sort(key=lambda corner: (-corner.x, corner.y))
    BR = temp[0]
    temp.sort(key=lambda corner: (-corner.x, -corner.y))
    TR = temp[0]

    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    
    #order of corners is TR->TL->BL->BR
    #TR->TL
    CornerArray.append(TR)
    temp.pop(0) 
    i = 0
    temp.sort(key=lambda corner: (-corner.y, -corner.x))
    while (temp[i].x != TL.x and temp[i].y != TL.y):
        CornerArray.append(temp[i])
        temp.pop(i)
        i += 1

    #TL->BL
    CornerArray.append(TL)
    temp.pop(0)
    i = 0
    temp.sort(key=lambda corner: (corner.x, -corner.y))
    while (temp[i].x != BL.x and temp[i].y != BL.y):
        CornerArray.append(temp[i])
        temp.pop(i)
        i += 1

    #BL->BR
    CornerArray.append(BL)
    temp.pop(0)
    i = 0
    temp.sort(key=lambda corner: (corner.y, -corner.x))
    while (temp[i].x != BR.x and temp[i].y != BR.y):
        CornerArray.append(temp[i])
        temp.pop(i)
    
    #Rest is Coordinates from BR to TR, sort by largest x and smallest y than just add
    CornerArray.append(BR)
    temp.pop(0)
    temp.sort(key=lambda corner: (-corner.x, corner.y))
    if (len(temp) != 0):
        for i in range (len(temp)):
            CornerArray.append(temp[i])
    
    #CornerArray is now a sorted array of corners from TR->TL->BL->BR order


def discretizePoints(dx):
    #Use linear equation to map points and then break them up into discrtized poitns
    #dx is how many steps or size of discretized points
    #Run between all vertices = # of times to loop and fill out points is num points - 1
    pointArrayX.append(CornerArray[0].x)
    pointArrayY.append(CornerArray[0].y)
    for i in range (len(CornerArray)):
        if (i == len(CornerArray) - 1):
            step = (CornerArray[0].x - CornerArray[i].x) / dx
        else:
            step = (CornerArray[i+1].x - CornerArray[i].x) / dx
        
        if (step == 0): # Vertical Line
            if (i == len(CornerArray) - 1):
                Vertstep = (CornerArray[0].y - CornerArray[i].y) / dx
            else:
                Vertstep = (CornerArray[i+1].y - CornerArray[i].y) / dx
                
            for j in range(dx):
                pointArrayX.append(CornerArray[i].x)
                pointArrayY.append(CornerArray[i].y + Vertstep*(j+1))
        else: 
            for j in range(dx):
                pointArrayX.append(CornerArray[i].x + step*(j+1))

            if (i == len(CornerArray) - 1):
                slope = (CornerArray[0].y - CornerArray[i].y) / (CornerArray[0].x - CornerArray[i].x)
            else:
                slope = (CornerArray[i+1].y - CornerArray[i].y) / (CornerArray[i+1].x - CornerArray[i].x)
                    
            for k in range(dx):
                pointArrayY.append(slope * (pointArrayX[k + i*dx] - CornerArray[i].x) + CornerArray[i].y)
                  

def DFT(x):
    X = []
    for i in range (len(x[0])):
        X.append((x[0][i] + x[1][i]*1j)* np.exp(-1 * 1j * i))
    return X      

def DTFS(x):
    X = []
    for i in range(len(x)):
        component = 0;
        for k in range(len(x)):
            component += np.exp(k * 1j * i)
        X.append(component)
    return np.real(X), np.imag(X)

    
def main():
    readPic("TAMU.png")
    discretizePoints(10) 
    FSY = DTFS(pointArrayY)
    FSX = DTFS(pointArrayX)
    X = DFT(FSX)
    Y = DFT(FSY)
    
    Sum = 0
    for i in range(len(X)):
        Sum += np.imag(X[i])**2 + np.real(X[i])**2
    AmplitudeX = np.sqrt(Sum)
    Sum = 0
    for i in range(len(Y)):
        Sum += np.imag(Y[i])**2 + np.real(Y[i])**2
    AmplitudeY = np.sqrt(Sum)
    Frequency = (2 * np.pi) / 1
    
    print(X)
    print()
    print(Y)
    plt.plot(X,Y)
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes(xlim = (-AmplitudeX, AmplitudeX), ylim = (-1e-16, 1e-12))


    line, = ax.plot([], [], lw=2)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        FTX = AmplitudeX * np.cos(Frequency * i) 
        FTY = AmplitudeY * np.sin(Frequency * i)
        line.set_data(FTX, FTY)
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 100, interval=50, blit=True)
    anim.save('test.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
     

        
        