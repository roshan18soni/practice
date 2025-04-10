""" 
Largest size square submatrix with all 1s
Approach: Dynamic Programing
https://www.youtube.com/watch?v=auS1fynpnjo&ab_channel=takeUforward start at 7:48
O(m*n)/O(m*n)
 """
def largestSquareSubmatrix(matrix):
    if not matrix or not matrix[0]:
        return 0
    
    # Dimensions of the matrix
    rows = len(matrix)
    cols = len(matrix[0])
    
    # Create a DP table initialized with zeros
    dp = [[0] * cols for _ in range(rows)]
    ######DO NOT USE#### 
    # dp=[[0]*num_cols]*num_rows
    # This line creates a shallow copy of lists in Python, meaning each row in DP will reference 
    # the same list in memory. As a result, updating one element of dp[i][j] will modify all rows in the same column. 
    # This causes incorrect behavior when updating dp.
    
    
    max_side = 0  # To track the maximum size of the square sub-matrix
    
    # Fill the dp table
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1  # Base case for the first row/column
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                # Update the maximum size found so far
                max_side = max(max_side, dp[i][j])
    
    # The area of the largest square sub-matrix is max_side^2
    return max_side * max_side

""" 
Rotate a Rectangular Image by 90 Degree Clockwise
Approach: traspose then reverse rows
O(m*n)/O(m*n)
 """
def rotageBy90Degree(matrix):
    rows= len(matrix)
    cols= len(matrix[0])
    transposed_matrix= [[0]*rows for _ in range(cols)]

    #interchange rows and columns
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i]= matrix[i][j]

    #reverse each row
    rows= len(transposed_matrix)
    cols= len(transposed_matrix[0])

    for i in range(rows):
        for j in range(cols//2):
            transposed_matrix[i][j], transposed_matrix[i][cols-1-j]= transposed_matrix[i][cols-1-j], transposed_matrix[i][j]

    return transposed_matrix

""" 
Search in a row wise and column wise sorted matrix
O(m+n)/O(1)
example of worst case time complexity m+n is: target as botton left element.
 """
def searchMatrix(matrix, target):
    if not matrix or not matrix[0]:
        return False  # Edge case for empty matrix
    
    # Dimensions of the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    # Start at the top-right corner
    row, col = 0, num_cols - 1
    
    while row < num_rows and col >= 0:
        if matrix[row][col] == target:
            return (row, col)  # Found the target, return the position
        elif target< matrix[row][col]:
            col -= 1  # Move left, as the current element is too large
        else:
            row += 1  # Move down, as the current element is too small
    
    return False  # If we exit the loop, the target was not found

""" 
Print a given matrix in spiral form
O(m*n)/O(1)
 """        
def spiralOrder(matrix):
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse from left to right on the top row
        for i in range(left, right + 1):
            print(matrix[top][i])
        top += 1
        
        # Traverse from top to bottom on the right column
        for i in range(top, bottom + 1):
            print(matrix[i][right])
        right -= 1
        
        if top <= bottom:
            # Traverse from right to left on the bottom row
            for i in range(right, left - 1, -1):
                print(matrix[bottom][i])
            bottom -= 1
        
        if left <= right:
            # Traverse from bottom to top on the left column
            for i in range(bottom, top - 1, -1):
                print(matrix[i][left])
            left += 1
""" 
Given a boolean matrix mat[M][N] of size M X N, modify it such that if a matrix cell mat[i][j] is 1 (or true) 
then make all the cells of ith row and jth column as 1
O(m*n)/O(m+n)
 """
def booleanMatrix(matrix):

    num_rows= len(matrix)
    num_cols= len(matrix[0])

    rows_arr= [0]* len(matrix)
    cols_arr= [0]* len(matrix[0])

    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j]==1:
                rows_arr[i]=1
                cols_arr[j]=1

    for i in range(num_rows):
        for j in range(num_cols):
            if rows_arr[i]==1 or cols_arr[j]==1:
                matrix[i][j]=1

    return matrix

""" 
Min Cost Path-
Given a cost matrix cost[][] and a position (M, N) in cost[][], write a function that returns cost of minimum cost 
path to reach (M, N) from (0, 0). Each cell of the matrix represents a cost to traverse through that cell. 
The total cost of a path to reach (M, N) is the sum of all the costs on that path (including both source and destination). 
You can only traverse down, right and diagonally lower cells from a given cell, 
i.e., from a given cell (i, j), cells (i+1, j), (i, j+1), and (i+1, j+1) can be traversed. 
O(m*n)/O(m*n)
 """
def min_cost_path(cost):
    m, n = len(cost), len(cost[0])
    dp = [[0] * n for _ in range(m)]

    dp[0][0] = cost[0][0]

    # Fill first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + cost[0][j]

    # Fill first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + cost[i][0]

    # Fill the rest
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = cost[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]

""" 
Given a binary 2D array, where each row is sorted. Find the row with the maximum number of 1s. 
O(m+n)/O(1)
 """
def rowWithMaxOnes(mat):
    M = len(mat)       # Number of rows
    N = len(mat[0])    # Number of columns

    max_row_index = -1 # Index of the row with the maximum number of 1's
    max_ones_count=0
    j = N - 1          # Start from the top-right corner of the matrix
    
    # Traverse each row starting from the top-right corner
    for i in range(M):
        # Move left until a 0 is found or we run out of columns
        while j >= 0 and mat[i][j] == 1:
            j -= 1    # Move left
            max_ones_count+=1
            max_row_index = i  # Update the row index

    return max_row_index, max_ones_count

""" 
Find Number of Islands in matrix using DFS
O(m*n)/O(m*n)
 """
def numIslands_dfs(grid):
    if not grid:
        return 0

    def dfs(grid, i, j):
        nonlocal islands
        nonlocal size_of_islands
        # Check if the current position is out of bounds or if it is water (0)
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
            return
        
        # Mark the current cell as visited by setting it to '0'
        grid[i][j] = 0
        size_of_islands[islands]= size_of_islands.get(islands, 0)+1
        
        # Perform DFS on the 8 adjacent cells
        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                dfs(grid, i+row_offset, j+col_offset)


    islands = 0
    size_of_islands={}

    # Iterate through each cell in the grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                # If a cell contains '1', it's the start of a new island
                islands += 1
                # Use DFS to mark the entire island
                dfs(grid, i, j)

    return islands, size_of_islands

""" 
Find Number of Islands in matrix using BFS
O(m*n)/O(m*n)
 """
from collections import deque
def numIslands_bfs(grid):

    def bfs(grid, i, j):
        nonlocal num_islands
        nonlocal size_of_islands

        rows= len(grid)
        cols= len(grid[0])
        myQ= deque()
        myQ.append((i, j))
        grid[x][y]=0 #mark visited as soon as appended to avoid duplicates during neighbour finding

        while myQ:
            x, y= myQ.popleft()
            size_of_islands[num_islands]= size_of_islands.get(num_islands, 0)+1
            # Perform BFS on the 8 adjacent cells
            for row_offset in range(-1, 2):
                for col_offset in range(-1, 2):
                    if row_offset == 0 and col_offset == 0:
                        continue  # Skip the center cell
                    nx, ny= x+row_offset, y+col_offset
                    if nx>=0 and nx<rows and ny>=0 and ny<cols and grid[nx][ny]==1:
                        myQ.append((nx, ny))
                        grid[nx][ny]=0 #mark visited as soon as appended to avoid duplicates during neighbour finding
                
    rows= len(grid)
    cols= len(grid[0])

    num_islands=0
    size_of_islands={}
    for i in range(rows):
        for j in range(cols):
            if grid[i][j]==1:
                num_islands+=1
                bfs(grid, i, j)

    return num_islands, size_of_islands

""" 
find max sum with contiguous sub array
Kadane algo
O(n)/(1)
 """
def maxSubarraySum(arr):
    current_sum=0
    max_sum=0
    subarr_start_index=0
    subarr_end_index=0

    for i in range(len(arr)):
        # if adding current_sum(sum till previous ele) to current element making it smaller than itself 
        # then start the current sum from current element
        current_sum= max(current_sum+arr[i], arr[i])

        if current_sum==arr[i]:
            subarr_start_index=i

        max_sum= max(max_sum, current_sum)

        if current_sum==max_sum:
            subarr_end_index=i

    return max_sum, subarr_start_index, subarr_end_index

""" 
Maximum sum rectangle in a 2D matrix
Approach: merging the rows(sum) to convert them to 1D array and apply Kadane's algo
O(m*m*n)/O(n)
 """
def maximumSumRectangle(matrix):

    rows= len(matrix)
    cols= len(matrix[0])
    col_start_index=0
    col_end_index=0
    row_start_index=0
    row_end_index=0

    max_sum= float('-inf')
    for start_row in range(rows):
        temp= [0]* cols

        for end_row in range(start_row, rows):
            for col in range(cols):
                temp[col]+= matrix[end_row][col]

            response= maxSubarraySum(temp)
            current_max_sum= response[0]
            
            max_sum= max(max_sum, current_max_sum)
            if current_max_sum==max_sum:
                col_start_index= response[1]
                col_end_index= response[2]
                row_start_index=start_row
                row_end_index=end_row

    return max_sum, col_start_index, col_end_index, row_start_index, row_end_index


""" 
Rotate Matrix Clockwise by 1
O(m*n)/O(1)
 """
def rotateMatrixByOne(mat):

    if not len(mat):
        return

    """
        top : starting row index
        bottom : ending row index
        left : starting column index
        right : ending column index
    """

    top = 0
    bottom = len(mat)-1

    left = 0
    right = len(mat[0])-1

    while left < right and top < bottom:

        # Store the first element of next row,
        # this element will replace first element of
        # current row
        prev = mat[top+1][left]

        # Move elements of top row one step right
        for i in range(left, right+1):
            curr = mat[top][i]
            mat[top][i] = prev
            prev = curr

        top += 1

        # Move elements of rightmost column one step downwards
        for i in range(top, bottom+1):
            curr = mat[i][right]
            mat[i][right] = prev
            prev = curr

        right -= 1

        # Move elements of bottom row one step left
        for i in range(right, left-1, -1):
            curr = mat[bottom][i]
            mat[bottom][i] = prev
            prev = curr

        bottom -= 1

        # Move elements of leftmost column one step upwards
        for i in range(bottom, top-1, -1):
            curr = mat[i][left]
            mat[i][left] = prev
            prev = curr

        left += 1

""" 
Given a Boolean Matrix, find k such that all elements in k'th row are 0 and k'th column are 1.
Approach:
There can be at most one k that can be qualified to be an answer (Why? Note that if k'th row has all 0's probably except mat[k][k], then no column can have all 1')s. 
If we traverse the given matrix from a corner (preferably from top right and bottom left), we can quickly discard complete row or complete column based on below rules. 
If mat[i][j] is 0 and i != j, then column j cannot be the solution. 
If mat[i][j] is 1 and i != j, then row i cannot be the solution.

O(m+n)/O(1)
 """    
def find_k(matrix):
    if not matrix or not matrix[0]:
        return -1  # Handle empty matrix

    rows = len(matrix)
    cols = len(matrix[0])

    if rows != cols:
        return -1  # Handle non-square matrix

    for row in matrix:
        if any(cell not in (0, 1) for cell in row):
            raise ValueError("Matrix contains invalid data. Only 0 and 1 are allowed.")

    def check_for_potential_k(k, matrix):
        rows = len(matrix)
        cols = len(matrix[0])

        for col in range(0, cols):
            if matrix[k][col] != 0 and col != k:
                return False
        for row in range(0, rows):
            if matrix[row][k] != 1 and row != k:
                return False

        return True

    # Start from top right
    start_col_index = cols - 1
    for i in range(0, rows):
        while start_col_index >= 0:
            if matrix[i][start_col_index] == 0:
                if i == start_col_index:
                    if check_for_potential_k(i, matrix):
                        return i
                    else:
                        return -1
                else:
                    # Moving to previous column
                    start_col_index -= 1
            else:
                if i == start_col_index:
                    if check_for_potential_k(i, matrix):
                        return i
                    else:
                        return -1
                else:
                    # Moving to next row
                    break

    return -1  # If no valid k is found

""" 
Largest Rectangle in hitogram
O(n)/O(n)
This solution is: https://www.youtube.com/watch?v=X0X6G-eWgQ8&ab_channel=takeUforward
Try understanding more optimized one: https://www.youtube.com/watch?v=jC_cWLy7jSI&ab_channel=takeUforward
 """
def largestRectangleInHistogram(arr):
    my_stack=[]
    left_small=[0]*len(arr)
    right_small=[0]*len(arr)
    max_area=0

    #find closest left smaller
    for i in range(len(arr)):
        while my_stack and arr[my_stack[-1]]>=arr[i]:
            my_stack.pop()

        if my_stack:
            left_small[i]=my_stack[-1]+1
        else:
            left_small[i]=0
            
        my_stack.append(i)

    #clear stack to reuse
    my_stack.clear

    #find closest right smaller
    for i in range(len(arr)-1, -1, -1):
        while my_stack and arr[my_stack[-1]]>=arr[i]:
            my_stack.pop()

        if my_stack:
            right_small[i]=my_stack[-1]-1
        else:
            right_small[i]=len(arr)-1
            
        my_stack.append(i)

    #find max area
    for i in range(len(arr)):
        max_area= max(max_area, (right_small[i]-left_small[i]+1)*arr[i])
            
    return max_area   
 
""" 
Maximum size rectangle binary sub-matrix with all 1s
O(m*n)/O(n)
 """
def maxRectangle(matrix):
    cell_hieght=[0]*len(matrix)
    rows= len(matrix)
    cols= len(matrix[0])
    max_rectangle_area=0

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]==1:
                cell_hieght[j]+=1
            else:
                cell_hieght[j]=0

        print(cell_hieght)
        max_area_till_this_row= largestRectangleInHistogram(cell_hieght)
        print(max_area_till_this_row)
        max_rectangle_area= max(max_rectangle_area, max_area_till_this_row)
        
    return max_rectangle_area

# arr=[2, 1, 5, 3, 2, 3, 1]
# print(largestRectangleInHistogram(arr))

# A = [
#     [0, 1, 1, 0],
#     [1, 1, 1, 1],
#     [1, 1, 1, 1],
#     [1, 1, 0, 0]
#     ]

# print(maxRectangle(A))

A = [
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1]
    ]

print(numIslands_dfs(A))
# print(numIslands_bfs(A))