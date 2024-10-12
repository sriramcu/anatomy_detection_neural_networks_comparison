"""
Program to understand how to extract rows and columns from a np array representing an image.
Logic is used in edge removal in the custom preprocessing function
"""
import numpy as np

arr = np.array([
[
[0,0,0,0], # Row 0, column 0
[1,1,1,1], # Row 0, column 1
],
[
[3,3,3,3],  # Row 1, column 0
[4,4,4,4],  # Row 1, column 1
],
[
[6,6,6,6],
[8,6,7,7],
],
])

print(arr.shape)

print(arr[:,0,:]) # First Column
print(arr[:,1,:]) # Second Column

print(arr[0,:,:]) # First Row
print(arr[1,:,:]) # Second Row
print(arr[2,:,:]) # Third Row

print(arr[[0,2],:,:]) # First and Third Row

print(np.mean(arr, axis=0))  
# The average is taken over the flattened array by default, otherwise over the specified axis. 