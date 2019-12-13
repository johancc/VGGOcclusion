from PIL import Image
import os
import random
import math

cwd = os.getcwd() + "/"
imgPath = cwd + "cars_train/" 
shapePath = cwd + "objects/"
finalimgPath = cwd + "trial_imgs/test/"

def composite(imFile1, imFile2): #put im1 on top of im2, v1
    try:
        obj = Image.open(imFile1)
    except:
        print("issue opening " + imFile1)
        return

    try:
        bkgrd = Image.open(imFile2)
    except:
        print("issue opening " + imFile2)
        return

    bkgrd = bkgrd.resize((224, 224))
    obj = obj.resize((75, 75))
    objdata = obj.getdata()
    bkgrddata = bkgrd.getdata()
    finalimgdata = []

    for i in range(224): #y
        for j in range(224): #x
            appendBkgrd = True
            if (i >= 74 and i < 149) and (j >= 74 and j < 149): #inside the overlapping img
                x = j-74
                y = i-74
                pos = y*75+x
		#if the pixel val in the obj img is not near (255,255,255), i.e. is not white, append the obj img's pixel
		#(bc the background of the obj imgs is white)
		#some of the objs are white-ish so i lowered the threshold
                if objdata[pos][0] < 253 and objdata[pos][1] < 253 and objdata[pos][2] < 253: 
                    appendBkgrd = False
                    
            if appendBkgrd:
                pos = 224*i+j
                finalimgdata.append(bkgrddata[pos])
            else:
                x = j-74
                y = i-74
                pos = y*75+x 
                finalimgdata.append(objdata[pos])

    finalimg = Image.new('RGB', (224, 224))
    finalimg.putdata(finalimgdata)
    return finalimg

def composite2(imFile1, imFile2, objSize, bkgrdSize): #put im1 on top of im2, v2
    try:
        obj = Image.open(imFile1)
    except:
        print("issue opening " + imFile1)
        return

    try:
        bkgrd = Image.open(imFile2)
    except:
        print("issue opening " + imFile2)
        return

    bkgrd = bkgrd.resize((bkgrdSize, bkgrdSize))
    obj = obj.resize((objSize, objSize))
    objdata = obj.getdata()
    bkgrddata = bkgrd.getdata()
    finalimgdata = []

    for i in range(bkgrdSize): #y
        for j in range(bkgrdSize): #x
            appendBkgrd = True
            left = (bkgrdSize-objSize)//2
            if (i >= left and i < left + objSize) and (j >= left and j < left + objSize): #inside the overlapping img
                x = j-left
                y = i-left
                pos = y*objSize+x
		#if the pixel val in the obj img is not near (255,255,255), i.e. is not white, append the obj img's pixel
		#(bc the background of the obj imgs is white)
		#some of the objs are white-ish so i lowered the threshold
                if objdata[pos][0] < 253 and objdata[pos][1] < 253 and objdata[pos][2] < 253: 
                    appendBkgrd = False
                    
            if appendBkgrd:
                pos = bkgrdSize*i+j
                finalimgdata.append(bkgrddata[pos])
            else:
                x = j-left
                y = i-left
                pos = y*objSize+x
                finalimgdata.append(objdata[pos])

    finalimg = Image.new('RGB', (bkgrdSize, bkgrdSize))
    finalimg.putdata(finalimgdata)
    return finalimg

def makeAllImages(dirPath1, dirPath2, dirPath3, finalFilename, bkgrdSize, n):
    dir1gen = traverseDir(dirPath1) #generator for obj images
    dir2gen = traverseDir(dirPath2) #generator for background/car images
    
    for i in range(n):
        path1 = dirPath1 + next(dir1gen)
        path2 = dirPath2 + next(dir2gen)

        size = random.choice([0.4, 0.5, 0.6])
        objSize = int(math.pow(size, 0.5)*bkgrdSize)
        #print(objSize)
        img = composite2(path1, path2, objSize, bkgrdSize)
        if img is not None:
            img.save(dirPath3 + finalFilename + str(i) + ".jpg")

def traverseDir(dirPath):
    directory = os.fsencode(dirPath)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        yield filename
        

#trial = composite(imgPath+"000000000139.jpg", imgPath+"000000000285.jpg")
#trial = composite(shapePath+"0.42_703204_sk_lg.jpg", imgPath+"car3.jpg")
#trial.show()

makeAllImages(shapePath, imgPath, finalimgPath, "", 224, 1)
