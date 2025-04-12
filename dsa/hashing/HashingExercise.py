""" 
find subarray with zero sum
O(n)/O(n)
 """
def findSubArrayWithZeroSum(arr):
    sum_seen= set()

    sum=0
    for e in arr:
        sum+=e
        if sum==0 or sum in sum_seen:
            return True
        
        sum_seen.add(sum)

    return False

class TreeNode:
    def __init__(self, data) -> None:
        self.data= data
        self.left=None
        self.right=None

"""
Print tree vertically
O(n)/O(n) 
 """
def vertical_print(root:TreeNode):
    if root is None:
            return

    hd_map = {}

    def verticalPrintUtil(root:TreeNode, horizontalDistFromRoot=0):
        if root is None:
            return
        
        list= hd_map.get(horizontalDistFromRoot, [])
        list.append(root.data)
        hd_map[horizontalDistFromRoot]= list

        verticalPrintUtil(root.left, horizontalDistFromRoot-1)

        verticalPrintUtil(root.right, horizontalDistFromRoot+1)

    verticalPrintUtil(root, 0)

    for hd in sorted(hd_map):
        for e in hd_map[hd]:
            print(e, end=" ")
        print("")

""" 
Design a data structure that supports insert, delete, search and getRandom in constant time
O(1)/O(n)
 """
import random
class SpecialDataStructure:
    def __init__(self) -> None:
        self.myarr= []
        self.mymap= {}
        self.size=0

    def insert(self, data):
        self.myarr.append(data)
        self.size+=1
        self.mymap[data]= self.size-1

    def search(self, data):
        if data in self.mymap:
            return True
        return False
    
    def delete(self, data):
        if data not in self.mymap:
            return 'element not present'
        
        index= self.mymap[data]
        last_ele= self.myarr[-1]
        self.myarr[index]= last_ele
        self.mymap[last_ele]=index
        self.myarr.pop()
        del self.mymap[data]
        self.size-=1    

    def getRandom(self):
        random_index = random.randint(0, self.size - 1)
        return self.myarr[random_index]

""" 
Find itinerary
O(n)/O(n)
 """
def findItinerary(d:dict):
    reverse_d={}
    for k,v in d.items():
        reverse_d[v]=k

    start_point=''
    for k in d.keys():
        if k not in reverse_d:
            start_point=k
            break

    while start_point in d:
        print(f'{start_point} -> {d[start_point]}')
        start_point= d[start_point]

""" 
Longest Subarray with 0 Sum
O(n)/O(n)
 """
def largestZeroSumSubArray(arr):
    sum_seen={0: -1}
    sum=0
    max_len=0
    for i in range(len(arr)):
        sum+=arr[i]
        if sum in sum_seen:
            max_len= max(max_len, i-sum_seen[sum])
        else:
            sum_seen[sum]=i

    return max_len

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right.left = TreeNode(6)
# root.right.right = TreeNode(7)
# root.right.left.right = TreeNode(8)
# root.right.right.right = TreeNode(9)
# vertical_print(root)

# ds= SpecialDataStructure()
# ds.insert(10)
# ds.insert(20)
# ds.insert(30)
# ds.insert(40)
# print(ds.search(30))
# ds.delete(40)
# ds.insert(50)
# print(ds.search(50))
# print(ds.getRandom())

# d={}
# d['Chennai']= 'Banglore'
# d['Bombay']= 'Delhi'
# d['Goa']= 'Chennai'
# d['Delhi']= 'Goa'

# findItinerary(d)

arr = [15, -2, 2, -8, 1, 7, 10, 23]
print(largestZeroSumSubArray(arr))
