#----------------------------------------------------------
# Feature Map + ReLU + Max Pooling
# Tasks : 
# 1. Create a feature map with positive and negatives values
# 2. Apply ReLU
# 3. Apply 2x2 Max Pooling
# 4. Dsiplay output aftre each step 
# 5. Understand why pooling reduces the size
#---------------------------------------------------------
import numpy as np
#-------------------------------------------
# Step 1 : Crete a Feature Map
#-------------------------------------------
def create_feature_map():
    feature_map = np.array([
        [3, 3, 3],
        [0, 0, 0],
        [-3, -3, -3]
    ])

    print("\nOriginal Feature Map : ")
    print(feature_map)

    return feature_map
#-------------------------------------------
# Step 2 : Apply ReLU
# ReLU(x) = max(0,x)
# Negative values become zero
# Positive values remains unchanged
#-------------------------------------------
def ReLU():
    feature_map = create_feature_map()

    relu_output = np.maximum(0,feature_map)

    print("\n After ReLU :")
    print(relu_output)

    return relu_output
#-------------------------------------------
# Step 3 : Apply 2x2 Max Pooling
#-------------------------------------------
def Max_Pooling():
    feature_map = ReLU()
    rows,columns = feature_map.shape

    pool_size = 2
    output_rows = (rows - pool_size)//pool_size + 1
    output_columns = (columns -pool_size) //pool_size + 1

    pooled_output = np.zeros(
        (output_rows,output_columns),
        dtype=int
    )

    for i in range(output_rows):
        for j in range(output_columns):

            #Extract 2x2 region
            region = feature_map[
                i * pool_size : i * pool_size + pool_size,
                j * pool_size : j * pool_size + pool_size
            ]

            #find maximum value
            maximum = np.max(region)

            #store maximum value
            pooled_output[i][j] = maximum

    print("\n After 2x2 Max Pooling : ")
    print(pooled_output)

    return pooled_output
#--------------------------------------
# Main Function
#--------------------------------------
def main():
    pooled_output = Max_Pooling()

    print("Pooled Output : ")
    print(pooled_output)
#--------------------------------------
# Entry point of code
#--------------------------------------
if __name__ == "__main__":
    main()