from PIL import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
def main():
    top = tk.Tk() #creates window

    top.filename =filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))

    img = ImageTk.PhotoImage(Image.open(top.filename)) # image
    # img = Image.open('numbers.PNG')
    img= Image.open(top.filename)
    img_gray = img.convert('L') # converts the image to grayscale image
    # img_bin = img.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    # img_gray.show()
    img.show()
    ONE = 150
    a = np.asarray(img_gray) #from PIL to np array
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin) # from np array to PIL format
    #a_bin = binary_image(100,100, ONE)   #creates a binary image
    im_label, colour_label, table = blob_coloring_8_connected(a_bin, ONE)
    # print (table)      max min points
    colouredImage = np2PIL_color(colour_label)
    colouredImage.show()

    rectangles = draw_rectangles(table, img)
    rectangles.show()

    moment_calculations(table,img)



def binary_image(nrow,ncol,Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow,ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10


    for i in range (50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i-20][90-i+1] = 1
        mask_lines[i-20][90-i+2] = 1
        mask_lines[i-20][90-i+3] = 1


    #mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
    #mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    #mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    #mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    #imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    #imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge

def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img

def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    im = np.zeros(shape=(nrow, ncol), dtype=int)
    a = np.zeros(shape=max_label, dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
                c = bim[i][j]
                l = bim[i][j - 1]
                u = bim[i - 1][j]
                label_u = im[i - 1][j]
                label_l = im[i][j - 1]
                label_ur = im[i+1][j-1]
                label_ul = im[i-1][j-1]

                im[i][j] = max_label
                if c == ONE:
                    min_label = min( label_u, label_l, label_ur, label_ul)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label  :
                            update_array(a, min_label, label_u)
                        if min_label != label_l and label_l != max_label  :
                            update_array(a, min_label, label_l)
                        if min_label != label_ur and label_ur != max_label :
                            update_array(a, min_label, label_ur)
                        if min_label != label_ul and label_ul != max_label :
                            update_array(a, min_label, label_ul)

                else :
                    im[i][j] = max_label
    # final reduction in label array
    for i in range(k+1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    #second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]


    counter = -1
    uniqueLabels = []
    for i in range(nrow):
        for j in range(ncol):
            if im[i][j] in uniqueLabels:
                pass
            else:
                uniqueLabels.append(im[i][j])
                counter = counter + 1
    print("Label number is", counter)
    print("Length of the Labels: ", len(uniqueLabels)-1)
    print(*uniqueLabels, sep=", ")
    #filling table with "label-min_i-min_j-max_i-max_j"

    #table [i][0] 0=Label
    #table [i][1] 1 = max i
    #table [i][2] 2 = min i
    #table [i][3] 3 = max j
    #table [i][4] 4 = min j         label-min_i-min_j-max_i-max_j
    table = np.zeros((len(uniqueLabels)+1, 5))

    for a in range(len(uniqueLabels)):
        table[a][0] = uniqueLabels[a]

    #print("table is",table)
    #print(int(table[len(uniqueLabels)-1][0])) #int parantezine almazsan 289.0 çıkıyo.

    for ind in range(len(uniqueLabels)):
        for i in range(nrow):
            for j in range(ncol):
                if im[i][j] == int(table[ind+1][0]):
                    if table[ind+1][1] < i or table[ind+1][1] == 0:
                        table[ind+1][1] = i
                    if table[ind+1][2] > i or table[ind+1][2] == 0:
                        table[ind+1][2] = i
                    if table[ind+1][3] < j or table[ind+1][3] == 0:
                        table[ind+1][3] = j
                    if table[ind+1][4] > j or table[ind+1][4] == 0:
                        table[ind+1][4] = j

    #print("last ver of table:",table)

    return im, color_im, table


def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else: #a[index] == lab_small
            break

    return
def draw_rectangles (arr, image):
    draw = ImageDraw.Draw(image)
    #min y max y min x max x
    # table [i][0] 0=Label
    # table [i][1] 1 = max i
    # table [i][2] 2 = min i
    # table [i][3] 3 = max j
    # table [i][4] 4 = min j
    for i in range(len(arr)-1):
        draw.rectangle([arr[i+1][4], arr[i+1][2], arr[i+1][3], arr[i+1][1]], width=1, outline="red")
    return image
def moment_calculations (arr,image):
    for i in range(len(arr)-2):

        #im = Image.fromarray(image, 'RGB')
        box = [arr[i+1][4]+1, arr[i+1][2]+1, arr[i+1][3]-1, arr[i+1][1]-1]
        #im = image.crop(list[i][2], list[i][4], list[i][1], list[i][3])
        cropped = image.crop(box)
        cropped = cropped.resize((22, 22))
        cropped.show()
        #im.show()
        for j in range(22):
            for k in range(22):
                img_gray =cropped.convert('L')  # converts the image to grayscale image
                ONE = 1
                a = np.asarray(img_gray)  # from PIL to np array
                a_bin = threshold(a, 100, ONE, 0)
                if j < len(a_bin)-1:
                    print(int(a_bin[j][k]),end=" ")
            print()
        moments(a_bin)

def moments (arr):
    rawMoment = [[0,0,0,0],[0,0,0],[0,0],[0]]
    for i in range(len(rawMoment)):
        for k in range(len(rawMoment[i])):
            for x in range(len(arr)):
                for y in range(0,22):
                    rawMoment[i][k]=rawMoment[i][k]+ pow(x,i)*pow(y,k) * arr[x][y]
    x0=rawMoment[1][0]/rawMoment[0][0]
    y0=rawMoment[0][1]/rawMoment[0][0]
    print("Raw Moments are : ",rawMoment[0])
    print("                 ",rawMoment[1])
    print("                 ",rawMoment[2])
    print("                 ",rawMoment[3])
    print()
    centralMoments=[[0,0,0,0],[0,0,0],[0,0],[0]]
    for i in range(len(rawMoment)):
        for k in  range(len(rawMoment[i])):
            for x in range(0,22):
                for y in range(0,22):
                    centralMoments[i][k] = centralMoments[i][k] + pow((x-x0),i)* pow((y-y0),k) * arr[x][y]
    print("Central Moments are : ",centralMoments[0])
    print("                      ",centralMoments[1])
    print("                      ",centralMoments[2])
    print("                      ",centralMoments[3])
    normalizedCentralMoments = [[0, 0, 0, 0], [0, 0, 0], [0, 0], [0]]
    print()
    for i in range(len(rawMoment)):
        for k in  range(len(rawMoment[i])):
            normalizedCentralMoments[i][k]= centralMoments[i][k]/pow(centralMoments[0][0],((i+k)/2)+1)
    print("Normalized Moments are : ",normalizedCentralMoments[0])
    print("                       ",normalizedCentralMoments[1])
    print("                       ",normalizedCentralMoments[2])
    print("                       ",normalizedCentralMoments[3])
    print()
    H=[0,0,0,0,0,0,0]
    H[0]=normalizedCentralMoments[2][0]+normalizedCentralMoments[0][2]
    H[1]=pow(normalizedCentralMoments[2][0]-normalizedCentralMoments[0][2],2)+4*pow(normalizedCentralMoments[1][1],2)
    H[2]=pow(normalizedCentralMoments[3][0]-3*normalizedCentralMoments[1][2],2)+pow(3*normalizedCentralMoments[2][1]-normalizedCentralMoments[0][3],2)
    H[3]=pow(normalizedCentralMoments[3][0],2)+pow((normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3]),2)
    H[4]=(normalizedCentralMoments[3][0]-3*normalizedCentralMoments[1][2])*(normalizedCentralMoments[3][0]+ normalizedCentralMoments[1][2])*(pow(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2],2)-3*pow(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3],2))+(3*normalizedCentralMoments[2][1]-normalizedCentralMoments[0][3])*(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3])*(3*pow(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2],2)-pow(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3],2))
    H[5]=(normalizedCentralMoments[2][0]-normalizedCentralMoments[0][2])*(pow(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2],2)-pow(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3],2))+4*normalizedCentralMoments[1][1]*(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2])*(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3])
    H[6]=(3*normalizedCentralMoments[2][1]-normalizedCentralMoments[3][0])*(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2])*(pow(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2],2)-3*pow(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3],2))+(3*normalizedCentralMoments[1][2]-normalizedCentralMoments[3][0])*(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3])*(3*pow(normalizedCentralMoments[3][0]+normalizedCentralMoments[1][2],2)-pow(normalizedCentralMoments[2][1]+normalizedCentralMoments[0][3],2))
    print("Hu moments are : ", H)
    print()
    r_moments=[0,0,0,0,0,0,0,0,0,0]
    r_moments[0]=math.sqrt(H[1])/H[0]
    r_moments[1]=(H[0]+math.sqrt(H[1]))/(H[0]-math.sqrt(H[1]))
    r_moments[2]=math.sqrt(H[2])/math.sqrt(H[3])
    r_moments[3]=math.sqrt(H[2])/math.sqrt(abs(H[4]))
    r_moments[4]=math.sqrt(H[3])/math.sqrt(abs(H[4]))
    r_moments[5]=abs(H[5])/H[0]*H[2]
    r_moments[6]=abs(H[5])/H[0]*math.sqrt(abs(H[4]))
    r_moments[7]=abs(H[5])/H[2]*math.sqrt(H[1])
    r_moments[8]=abs(H[5])/math.sqrt(H[1]*abs(H[4]))
    r_moments[9]=abs(H[4])/H[2]*H[3]
    print("R moments are : ", r_moments)
    print()

    np.savetxt('raa',r_moments)
    database=np.loadtxt('raa')

# 00
# 01
# 02
# 03
#
# 10
# 11
# 12
#
# 20
# 21
#
# 30
if __name__=='__main__' :
    main()
