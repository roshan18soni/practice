
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

arr1 = [1, 5, 9, 10, 15, 20]
arr2=[2, 3, 8, 13]
mergeSortedArraysWithoutExtraSpae(arr1, arr2, len(arr1), len(arr2))

print(arr1)
print(arr2)
