""" 
Find 2 elements with given sum
Approach- One-pass hash table
Notes: Previous elements get added to dict, now check whether complement for next element exists in dict or not.
To just check whether such elements exists or not, Set is enough
Time: O(n)
Space: O(n)
"""

def twoSum(self, nums: list[int], target: int) -> list[int]:
    numMap = {}
    n = len(nums)

    for i in range(n):
        complement = target - nums[i]
        if complement in numMap:
            return [numMap[complement], i]
        numMap[nums[i]] = i

    return []  # No solution found


""" 
Majority Element- element appearing more than n/2 times
Approach- Hashmap
Note: Use dict for word count then iterate to find key with max count
Time: O(n)
Space: O(n)
 """

from collections import defaultdict 
def majorityElement(self, nums: list[int]) -> int:
    n = len(nums)
    m = defaultdict(int)
    
    for num in nums:
        m[num] += 1
    
    n = n // 2
    for key, value in m.items():
        if value > n:
            return key
    
    return 0

"""  
Find the Number Occurring Odd Number of Times
Approach- Hashmap
Time: O(n)
Space: O(n)
"""
def getOddOccurrence(arr,size):
     
    Hash=dict()
 
    # Putting all elements into the HashMap 
    for i in range(size):
        Hash[arr[i]]=Hash.get(arr[i],0) + 1
    
    # Iterate through HashMap to check an element
    # occurring odd number of times and return it
    for i in Hash:

        if(Hash[i]% 2 != 0):
            return i
    return -1

""" 
Merge an array of size n into another array of size m+n
There are two sorted arrays. Second one is of size m+n containing only m elements. Another one is of size n and contains n elements.
Note: 
if placeholder values in second array are added at the last then start processing the arrays from last
and start placing bigger values from last index ie k=m+n-1
But if placeholder values in second array are added at the begining then start processing the arrays from begining 
and start placing smaller values from begining index ie k=0
This code for first case.
Time: O(m+n)
Space: O(1)
 """

def merge_arrays_inplace(arr1, arr2, n, m):
    # Index of last element in arr1
    i = n - 1
    # Index of last element in arr2
    j = m - 1
    # Index of last position in merged array
    k = n + m - 1

    # Merge arr1 and arr2, starting from the end
    while i >= 0 and j >= 0:
        if arr1[i] > arr2[j]:
            arr2[k] = arr1[i]
            i -= 1
        else:
            arr2[k] = arr2[j]
            j -= 1
        k -= 1

    # If there are remaining elements in arr1, copy them to arr2
    while i >= 0:
        arr2[k] = arr1[i]
        i -= 1
        k -= 1

""" 
Merge two sorted arrays and return new array
Time: O(m+n)
Space: O(m+n)
 """
def merge_arrays_to_new(arr1, arr2, n, m):
    i=0
    j=0
    k=0
    new_array=[]
    while i<n and j<m:
        if arr1[i]<arr2[j]:
            new_array.append(arr1[i])
            i+=1
        else:
            new_array.append(arr2[j])
            j+=1
        k+=1

    #copying remaining elements
    if i<n:
        while i<n:
            new_array.append(arr1[i])
            i+=1
            k+=1
    if j<m:
        while j<m:
            new_array.append(arr2[i])
            j+=1
            k+=1


"""
Rotate an array 
Approach: reversal
Time: O(n)
Space: O(1)
 """
def reverse(arr, start, end):
    while start < end:
        arr[start], arr[end]= arr[end], arr[start]
        start+=1
        end-=1

def rotate_left(arr, d):
    if d==0:
        return
    n= len(arr)
    d= d%n

    reverse(arr, 0, d-1)
    reverse(arr, d, n-1)
    reverse(arr, 0, n-1)

def rotate_right(arr, d):
    if d==0:
        return
    n= len(arr)
    d= d%n

    reverse(arr, n-d, n-1)
    reverse(arr, 0, n-d-1)
    reverse(arr, 0, n-1)


""" 
Print Leaders: An element is a leader if it is greater than all the elements to its right side. And the rightmost element is always a leader. 
Time: O(n)
Space: O(1)
 """
def printLeaders(arr,size): 
    max=0
    for i in range(size-1, -1, -1):
        if arr[i]>max:
            max=arr[i]
            print(max, end=' ')

""" 
Check for Majority Element in a sorted array
Approach: (Using Linear Search): Linearly search for the first occurrence of the element, once you find it (let at index i), 
    check the element at index i + n/2. If the element is present at i+n/2 then return 1 else return 0.
Time: O(n)
Space: O(1)
 """
def isMajority(arr, n, x):
    # get last index according to n (even or odd) */
    last_index = (n//2 + 1) if n % 2 != 0 else (n//2)

    # search for first occurrence of x in arr[]*/
    for i in range(last_index):
        # check if x is present and is present more than n / 2 times */
        if arr[i] == x and arr[i + n//2] == x:
            return 1

""" 
Segregate 0s and 1s in an array
Approach: Two indexes to traverse
Time: O(n)
Space= O(1)
 """
def segregate(arr): 
    n= len(arr)
    left= 0
    right= n-1

    while left < right:
        while left< right and arr[left]==0:
            left+=1 
        while right> left and arr[right]==1:
            right-=1

        if left<right:
            arr[left], arr[right]= arr[right], arr[left]
            left+=1
            right-=1

""" 
Product of Array except itself
Approach: using prefix and suffix multiplication
Time: O(n)
Space: O(n)
 """
def getProd(arr): 
    n= len(arr)
    prefix= [1]*n
    suffix= [1]*n

    for i in range(1, n):
        prefix[i] = prefix[i-1]* arr[i-1]

    for i in range(n-2, -1, -1):
        suffix[i]= suffix[i+1] * arr[i+1]

    answer= [prefix[i] * suffix[i] for i in range(n)]
        
    return answer

""" 
Find the repeating elements in a given array, all elements of the array are in the range of 1 to N
Approach: using Array Elements as an Index
Time: O(n)
Space: O(1)
 """

def repeatingElements(arr): 
    n= len(arr)
    
    for i in range(len(arr)):
        ele_abs_value= abs(arr[i])
        if arr[ele_abs_value] < 0:
            print(abs(arr[i]), end= ' ')
        else:
            arr[arr[i]]= -arr[arr[i]]

""" 
Find the smallest missing number,
Given a sorted array of n distinct integers where each integer is in the range from 0 to m-1 and m > n. 
Find the smallest number that is missing from the array. 
Approach: Starting from arr[0] to arr[n-1] check until arr[i] != i. If the condition (arr[i] != i) is satisfied 
    then i is the smallest missing number
Time: O(n)
Space: O(1)
 """

def smallestMissingElement(arr, n, m): 
    
    for i in range(m):
        if i!= arr[i]:
            print(i, end=' ')
            break

""" 
Given an array arr[], find the maximum j - i such that arr[i] <= arr[j]
Given an array arr[] of N positive integers. The task is to find the maximum of j - i subjected to the constraint of arr[i] <= arr[j].
Time: O(n)
Space: O(n)
 """

def maxIndexDiff(arr, n):
     
    rightMax= [0]*n

    rightMax[n-1]= arr[n-1]

    for i in range(n-2, -1, -1):
        rightMax[i]= max(rightMax[i+1], arr[i])

    maxDiff= -1
    i, j=0, 0
    while i<n and j<n:
        if n-1-i <= maxDiff:
            return maxDiff
        if rightMax[j]>=arr[i]:
            maxDiff= max(maxDiff, j-i)
            j+=1
        else:
            i+=1

    return maxDiff

""" 
Find the first Subarray with given sum (Non-negative Numbers)
Approch: Slidding window
Time: O(n)
Space: O(1)
 """
def subArrayGivenSum(arr, n, sum):
    start, end=0, 0
    runningSum=0
    while start<n and end<n:
        print(f'start: {start}, end: {end}')
        runningSum+=arr[end]
        if runningSum<sum:
            end+=1
        elif runningSum>sum:
            runningSum-=arr[start]
            runningSum-=arr[end]
            start+=1
        elif runningSum==sum:
            print(f'Sum found between indexes {start} and {end}')
            return 1
            
    print("No subarray found")
    return 0

""" 
Find the smallest positive number missing from an unsorted array.
Approach1: Smallest positive number missing from an unsorted array by changing the input Array
        The idea is to mark the elements in the array which are greater than N and less than 1 with 1 to keep all numbers in a range of 1 to n.
Time: O(n)
Space: O(1)
Easier to understand
 """
def smallestMissingPositive_firstApproach(arr, n):
    if n==0:
        return 1
    if n==1 and arr[0]!=1:
        return 1
    
    # Check if 1 is present in array or not
    onePresent= False
    for ele in arr:
        if ele==1:
            onePresent=True
            break
    
    # If 1 is not present
    if onePresent==False:
        return 1

    # Changing values to 1
    for i in range(n):
        if arr[i]<1 or arr[i]>n:
            arr[i]=1

    # Updating indices according to values
    for i in range(n):
        val_at_index= abs(arr[i])
        if val_at_index!=1:
            arr[val_at_index-1]= -arr[val_at_index-1]

    # Finding which index has value less than n
    for i in range(1, n):
        if arr[i]>0:
            return i+1

    # If array has values from 1 to n
    return n+1

""" 
Approach: Smallest positive number missing from an unsorted array by Swapping,
        The idea is to swap the elements which are in the range 1 to N should be placed at their respective indexes.
Time: O(n)
Space: O(1)
 """
def smallestMissingPositive_SecondApproach(arr, n):

    # Loop to traverse the whole array
    for i in range(n):

        # Loop to check boundary
        # condition and for swapping
        while (arr[i] >= 1 and arr[i] <= n
               and arr[i] != arr[arr[i] - 1]):
            temp = arr[i]
            arr[i] = arr[arr[i] - 1]
            arr[temp - 1] = temp

    # Checking any element which
    # is not equal to i + 1
    print(arr)
    for i in range(n):
        print(arr[i])
        if (arr[i] != i + 1):
            return i + 1

    # Nothing is present return last index
    return n + 1

""" 
Find the two numbers with odd occurrences in an unsorted array
Approach: Using xor
https://www.youtube.com/watch?v=pnx5JA9LNM4&t=967s&ab_channel=Pepcoding
The 2's complement of a number is a way of representing negative numbers in binary
Time: O(n)
Space: O(n)
 """

def findTwoOddOcuurances(arr, n):
    xorAll=0
    for ele in arr:
        xorAll= xorAll^ele

    rightMostSetBitMask= xorAll & -(xorAll)

    firstGroupWithUnsetBit=0
    secondGroupWithSetBit=0

    for ele in arr:
        if ele & rightMostSetBitMask ==0:
            firstGroupWithUnsetBit= firstGroupWithUnsetBit^ ele
        else:
            secondGroupWithSetBit= secondGroupWithSetBit^ ele

    if firstGroupWithUnsetBit< secondGroupWithSetBit:
        print(firstGroupWithUnsetBit)
        print(secondGroupWithSetBit)
    else:
        print(secondGroupWithSetBit)
        print(firstGroupWithUnsetBit)
""" 
Largest subarray with equal number of 0s and 1s
Approach: Hashmap
Time: O(n)
Space: O(1)
 """
def findSubarray(arr, n):

    sumDict={0:-1}
    maxLen=0
    start=-1
    sum=0
    for i in range(n):
        if arr[i]==0:
            sum-=1
        else:
            sum+=1
        print(sum)
        if sum in sumDict:
            if i- sumDict[sum] > maxLen:
                maxLen= i-sumDict[sum]
                start= sumDict[sum]+1
        else:
            sumDict[sum]=i
        print(f'start {start}')
        print(sumDict)
        
    if maxLen>1:
        return start, start+maxLen-1
    else: 
        return 'na' 

""" 
Replace every element with the greatest element on right side
Approach: The idea is to start from the rightmost element, 
        move to the left side one by one, and keep track of the maximum element. Replace every element with the maximum element
Time: O(n)
Space: O(1)
 """
def replaceWithGreatestElementAtRight(arr, n):
    max=arr[n-1]
    arr[n-1]=-1
    for i in range(n-2, -1, -1):
        temp=arr[i]
        arr[i]=max
        if temp>max:
            max=temp

""" 
Replace every element with the smallest of all other array elements
Approach: The idea is to maintain prefix and suffix min arrays. Maintain leftMin[] and rightMin[] arrays which stores 
    the minimum on the left and right subarrays for every array element. Once computed, replace every ith index of the 
    original array by storing the minimum of leftMin[i] and rightMin[i].
Time: O(n)
Space: O(n)
 """
import sys
def replaceWithSmallestOfOtherElements(arr, n):
    leftSmallest= [0]*n
    rightSmallest= [0]*n

    leftSmallest[0]= sys.maxsize
    rightSmallest[n-1]= sys.maxsize

    for i in range(1, n):
        leftSmallest[i]= min(arr[i-1], leftSmallest[i-1])

    for i in range(n-2, -1, -1):
        rightSmallest[i]= min(arr[i+1], rightSmallest[i+1])

    for i in range(n):
        arr[i]= min(leftSmallest[i], rightSmallest[i])

""" 
Stock Buy Sell to Maximize Profit
Time: O(n)
Space: O(1)
 """
def stockBuySellForMaxProfit(arr, n):
    profit=0
    recentAction=None
    for i in range(1, n):
        if arr[i]>=arr[i-1]:
            profit+= (arr[i]-arr[i-1])
            if recentAction!='buy':
                recentAction='buy'
                print(f'Buy on day: {i-1}')
            if i==n-1:
                print(f'Sell on day: {i}')
        else:
            if recentAction=='buy':
                recentAction='sell'
                print(f'Sell on day: {i-1}')

    print(f'total profit: {profit}')

""" 
Find common elements in three sorted arrays
Approach: Three Pointer
Time: O(n)
Space: O(1)
 """
def commonElements(arr1, arr2, arr3, n1, n2, n3):
    i, j, k=0,0,0
    while i<n1 and j<n2 and k<n3:
        if arr1[i]==arr2[j]==arr3[k]:
            print(arr1[i])
            i+=1
            j+=1
            k+=1
        elif arr1[i]<arr2[j]:
            i+=1
        elif arr2[j]<arr3[k]:
            j+=1
        else:
            k+=1

""" 
Next/Previous Greater Element (NGE) and Next/Previous Smaller Element for every element in given Array
Approach: Using Stack
O(n)/O(n)
 """
def nextGreaterEle(arr, n):
    stack=[]
    result=[-1]*n
    for i in range(n-1, -1, -1):
        while stack:
            top_ele= stack[-1]
            if top_ele<=arr[i]:
                stack.pop()
            else:
                result[i]=top_ele
                break

        stack.append(arr[i])

    return result

def previousGreaterEle(arr, n):
    stack=[]
    result=[-1]*n
    for i in range(n):
        while stack:
            top_ele= stack[-1]
            if top_ele<=arr[i]:
                stack.pop()
            else:
                result[i]=top_ele
                break

        stack.append(arr[i])

    return result

def indexOfPreviousSmaller(arr, n):
    stack=[]
    result=[-1]*n #keeping index of previous smaller ele

    for i in range(n):
        while stack:
            val, index= stack[-1]
            if val>=arr[i]:
                stack.pop()
            else:
                result[i]= index
                break

        stack.append((arr[i], i))
    return result

def indexOfNextSmaller(arr, n):
    stack=[]
    result=[n]*n #keeping index of next smaller ele

    for i in range(n-1, -1, -1):
        while stack:
            val, index= stack[-1]
            if val>=arr[i]:
                stack.pop()
            else:
                result[i]=index
                break

        stack.append((arr[i], i))

    return result

""" 
Largest area of hitogram
Approach: prev and next smaller elements using stack
O(n)/O(n)
 """
def largetAreaOfHistogram(arr, n):
    
    pre_smaller_index=indexOfPreviousSmaller(arr, n)
    next_smaller_index=indexOfNextSmaller(arr, n)
    max_area=0
    for i in range(n):
        width= next_smaller_index[i]-pre_smaller_index[i]-1
        hight= arr[i]
        max_area= max(max_area, width*hight)

    return max_area
    
""" 
Trapping Rain Water
""" 
""" 
Approach 1: Left and right max
O(n)/O(n)
 """
def trappingRainWaterApproach1(arr, n):

    leftMaxArr=[-1]*n
    rightMaxArr=[-1]*n

    for i in range(1, n):
        leftMaxArr[i]= max(arr[i-1], leftMaxArr[i-1])

    for j in range(n-2, -1, -1):
        rightMaxArr[j]= max(arr[j+1], rightMaxArr[j+1])

    waterVolume=0
    for i in range(1, n-1):
        minOfHeightOfPreAndNextBuilding= min(leftMaxArr[i], rightMaxArr[i])
        if minOfHeightOfPreAndNextBuilding>arr[i]:
            waterVolume+=minOfHeightOfPreAndNextBuilding-arr[i]

    return waterVolume

"""
Approach 2: Two Pointer Approach
Time: O(n)
Space: O(1)
 """
def trappingRainWaterApproach2(arr, n):
    leftMax=arr[0]
    rightMax=arr[n-1]
    left=1
    right=n-2
    waterVolume=0

    while left<right:
        if leftMax<=rightMax:
            if leftMax>arr[left]:
                waterVolume+= leftMax-arr[left]
            leftMax= max(leftMax, arr[left])
            left+=1
        else:
            if rightMax>arr[right]:
                waterVolume+= rightMax-arr[right]
            rightMax= max(rightMax, arr[right])
            right-=1

    return waterVolume


""" 
Merge two sorted arrays with O(1) extra space
""" 
""" 
Approach: Two pointer
Time: O(min(N, M) + Nlog(N)+ Mlog(n))
Space: O(1)
 """
def mergeSortedArraysWithoutExtraSpae(arr1, arr2, n, m):
    i=n-1
    j=0

    while i>=0 and j<m:
        if arr1[i]> arr2[j]:
            arr1[i], arr2[j]= arr2[j], arr1[i]
            i-=1
            j+=1
        else:
            break

    arr1.sort()
    arr2.sort()

""" 
Using Gap method of Shell sort 
Time: O(log(m+n)*(m+n))
Space: O(1)
"""
def mergeArrays(a, b):
    n = len(a)
    m = len(b)
    gap = (n + m + 1) // 2

    while gap > 0:
        i = 0
        j = gap

        while j < n + m:
          
            # If both pointers are in the first array a[]
            if j < n and a[i] > a[j]:
                a[i], a[j] = a[j], a[i]
                
            # If first pointer is in a[] and 
            # the second pointer is in b[]
            elif i < n and j >= n and a[i] > b[j - n]:
                a[i], b[j - n] = b[j - n], a[i]
                
            # Both pointers are in the second array b
            elif i >= n and b[i - n] > b[j - n]:
                b[i - n], b[j - n] = b[j - n], b[i - n]
            i += 1
            j += 1

        # After operating for gap of 1 break the loop
        if gap == 1:
            break

        # Calculate the next gap
        gap = (gap + 1) // 2
