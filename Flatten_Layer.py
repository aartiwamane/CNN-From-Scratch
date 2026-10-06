#----------------------------------------------------
# Flatten Layer + Fully Connected Layer
# 
# Tasks : 
# 1. Take a 2D Matrix
# 2. COnvert it into 1D Vector
# 3. Pass it to Fully Connected Layer
# 4. Calculate FInal Output manually
# 5. Understand the role of FLatten Layer in CNN
#----------------------------------------------------
import numpy as np
#---------------------------------------------
# Step 1 : Create a 2D Feature Map
#---------------------------------------------
def create_feature_map():
    feature_map = np.array([
        [6, 4],
        [8, 6]
    ])
    print("\n Original Matrix : ")
    print(feature_map)

    return feature_map
#---------------------------------------------
# Step 2 : Apply Flatten Layer
#---------------------------------------------
def Flatten():
    feature_map = create_feature_map()
    flatten_output = feature_map.flatten()

    print("\n Output after flatten layer : ")
    print(flatten_output)

    return flatten_output
#---------------------------------------------
# Step 3 : Fully Ocnnected Layer
#---------------------------------------------
def Fully_connected_Layer():
    flatten_output = Flatten()

    #Manually selected Weights
    weights = np.array([0.5,0.2,0.1,0.4])

    #bias
    bias = 1

    print("\n Weights : ",weights)

    print("\n Bias : ",bias)

    #Calculate weighted sum 
    weighted_sum = np.sum(flatten_output*weights)

    #Calculate final output
    final_output = weighted_sum + bias

    return final_output

def main():
    final_output = Fully_connected_Layer()

    print("Final output from Fully connected layer : ",final_output)

if __name__ == "__main__":
    main()

