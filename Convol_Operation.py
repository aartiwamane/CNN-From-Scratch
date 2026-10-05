#---------------------------------------------
# 5*5 GrayScale Image 
# Top 2 Rows and Bottom 2 Rows are black (0)
# Middle Row is white (255)
# Apply 3*3 Kernel for Feature Map 
# Dispaly Feature Map
#---------------------------------------------
import numpy as np

def Create_Image_Kernel():
#---------------------------------------------
# Step 1 : Create 5x5 GrayScale Image
#---------------------------------------------
    image = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [1,1,1,1,1],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])

    print("\n Original 5x5 Image : ")
    print(image)
#------------------------------------------------------
# Step 2 : 3x3 Kernel for horizontal edge detection
#------------------------------------------------------
    kernel = np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ])
    print("\n 3x3 Kernel : ")
    print(kernel)

    return image,kernel
#---------------------------------------------
# Step 3 : Convol Operation
# Output size : (5-3+1)x(5-3+1) = 3x3
#---------------------------------------------
def Convol_Operation():
    image,kernel = Create_Image_Kernel()
    feature_map = np.zeros((3,3))

    for i in range(3):
        for j in range(3):

            #extract 3x3 region
            region = image[i:i+3,j:j+3]

            #Multiply and sum
            result = np.sum(region*kernel)

            #store result
            feature_map[i][j]= result

    return feature_map

def main():
    feature_map = Convol_Operation()

    print("\n Feature Map(Detected Edges) : ")
    print(feature_map)

if __name__ == "__main__":
    main()