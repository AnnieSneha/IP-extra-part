from PIL import Image<br>
from numpy import asarray<br>
img = Image.open('24.jpg')<br>
numpydata = asarray(img)<br>
print(numpydata)<br>

OUTPUT:<br>

[[[ 8 26 30]<br>
  [ 8 26 30]<br>
  [ 8 26 30]<br>
  ...
  [ 5 20 27]<br>
  [ 5 20 27]<br>
  [ 5 20 27]]<br>

 [[ 8 26 30]<br>
  [ 8 26 30]<br>
  [ 8 26 30]<br>
  ...
  [ 5 20 27]<br>
  [ 5 20 27]<br>
  [ 5 20 27]]<br>

 [[ 8 26 30]<br>
  [ 8 26 30]<br>
  [ 8 26 30]<br>
  ...
  [ 5 20 27]<br>
  [ 5 20 27]<br>
  [ 5 20 27]]<br>

 ...

 [[ 0  9 11]<br>
  [ 5 16 18]<br>
  [ 6 17 19]<br>
  ...
  [ 9 26 33]<br>
  [ 9 26 33]<br>
  [ 9 24 31]]<br>

 [[ 5 16 18]<br>
  [ 9 20 22]<br>
  [ 6 17 19]<br>
  ...
  [ 9 26 33]<br>
  [10 27 34]<br>
  [11 26 33]]<br>

 [[ 7 18 20]<br>
  [ 7 18 20]<br>
  [ 1 12 14]<br>
  ...<br>
  [10 27 34]<br>
  [12 29 36]<br>
  [13 28 35]]]<br>
  
  import numpy as np<br>
import matplotlib.pyplot as plt<br>

arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0, 0, 0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        #Find the distance to the center<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>

        #Make it on a scale from 0 to 1innerColor<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>

        #Calculate r, g, and b values
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        # print r, g, b<br>
        arr[y, x] = (int(r), int(g), int(b))<br>

plt.imshow(arr, cmap='gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/180199414-807f2c9e-d332-40c1-9eb0-49065766e767.png)

from PIL import Image<br>
import matplotlib.pyplot as plt<br>
  
#Create an image as input:<br>
input_image = Image.new(mode="RGB", size=(1000, 1000),color="pink")<br>
  
#save the image as "input.png"<br>
#(not mandatory)<br>
#input_image.save("input", format="png")<br>
  
#Extracting pixel map:<br>
pixel_map = input_image.load()<br>
  
#Extracting the width and height<br>
#of the image:<br>
width, height = input_image.size<br>
z = 100<br>
for i in range(width):
    for j in range(height):<br>
        
        #the following if part will create<br>
        #a square with color orange<br>
        if((i >= z and i <= width-z) and (j >= z and j <= height-z)):<br>
            
            #RGB value of orange.<br>
            pixel_map[i, j] = (230,230,250)<br>
  
        #the following else part will fill the<br>
        #rest part with color light salmon.<br>
        else:<br>
            
            #RGB value of light salmon.<br>
            pixel_map[i, j] = (216,191,216)<br>
  
#The following loop will create a cross<br>
#of color blue.<br>
for i in range(width):<br>
    
    # RGB value of Blue.<br>
    pixel_map[i, i] = (0, 0, 255)<br>
    pixel_map[i, width-i-1] = (0, 0, 255)<br>
  
#Saving the final output<br>
#as "output.png":<br>
#input_image.save("output", format="png")<br>
plt.imshow(input_image)<br>
plt.show()  <br>
#use input_image.show() to see the image on the<br>
#output screen.<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/180199536-48a0e2c4-dc68-460f-9780-4905bb774d57.png)


from PIL import Image:<br>
import numpy as np:<br>
w, h = 512, 512:<br>
data = np.zeros((h, w, 3), dtype=np.uint8):<br>
data[0:100, 0:100] = [255, 0, 0]:<br>
data[100:200, 100:200] = [255, 0, 255]:<br>
data[200:300, 200:300] = [0, 255, 0]:<br>
# red patch in upper left:<br>
img = Image.fromarray(data, 'RGB'):<br>
img.save('my.png'):<br>
plt.imshow(img):<br>
plt.show():<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/180201789-e09b2572-9b3c-4b0e-ac56-aea337ad3d32.png)

#MAX
import numpy as np<br>
# Create matrix<br>
matrix = np.array([[1, 2, 3],<br>
                   [4, 5, 6],<br>
                   [7, 8, 9]])<br>
# Return maximum element
np.max(matrix)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/181233563-bc1a51ae-8cfc-4145-a405-03af4372f016.png)

import numpy as np<br>
# Create matrix<br>
matrix = np.array([[1, 2, 3],<br>
                   [4, 5, 6],<br>
                   [7, 8, 9]])<br>
# Return maximum element
np.min(matrix)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/181233654-74294a94-9a16-40cb-91a5-6468c974c66f.png)

#Average
import imageio<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread("21.jpg")<br>
plt.imshow(img)<br>
np.average(img)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939284/181234309-6da80feb-9e7a-43e8-b7c2-1902bebb190a.png)<br>


import numpy as np<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
def printPattern(n):<br>

    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         
    # Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             
    # Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
#Driver Code<br>
n = 3;<br>
 
printPattern(n);<br>
w, h = n,n<br>
arraySize = np.zeros((h, w, 3))# dtype=np.uint8)<br>
arraySize[0:n, 0:n] = [255,0,0] # red patch in upper left<br>
#arraySize[0:n, 0:n] = [255, 0, 0]<br>
#arraySize[n:0,n:0] = [120,200,255]<br>
img = Image.fromarray(arraySize, 'RGB')<br>
plt.imshow(img)<br>
plt.show()<br>
#img.save('my.png')<br>
#img.show()<br>

![image](https://user-images.githubusercontent.com/97939284/181448195-1d1b405d-3c05-4bf4-895b-c335c7e7655f.png)



import matplotlib.pyplot as plt
M = ([1,2,3,4,5],
     [6,7,8,9,10],
     [11,12,13,14,15],
     [16,17,18,19,20] )    
plt.imshow(M)
plt.show()

![image](https://user-images.githubusercontent.com/97939284/181448367-75355d83-998a-40bd-bd13-b50dc0367ed4.png)

