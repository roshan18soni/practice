def maxSum(arr):
    maxi=float('-inf')
    sum=0
    for i in range(len(arr)):
        sum+=arr[i]
        maxi=max(maxi, sum)
        if sum<0:
            sum=0
