""" 
find k largest elements in an array
O(nlogk)/O(k)
 """
""" 
By implementing heapify function
 """
def find_k_largest(arr, k):
    
    # Function to heapify a subtree rooted at index i in an array of size n
    def heapify(arr, n, i):
        smallest = i  # Initialize smallest as root
        left = 2 * i + 1  # Left child index
        right = 2 * i + 2  # Right child index

        # If left child is smaller than root
        if left < n and arr[left] < arr[smallest]:
            smallest = left

        # If right child is smaller than smallest so far
        if right < n and arr[right] < arr[smallest]:
            smallest = right

        # If smallest is not root
        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]  # Swap
            heapify(arr, n, smallest)  # Recursively heapify the affected subtree

    # Function to build a min-heap from an array of size n
    def build_min_heap(arr, n):
        # Index of the last non-leaf node
        start_idx = n // 2 - 1

        # Perform reverse level order traversal from last non-leaf node and heapify each node
        for i in range(start_idx, -1, -1):
            heapify(arr, n, i)

    # Step 1: Build a min-heap with the first k elements
    heap = arr[:k]
    build_min_heap(heap, k)

    # Step 2: Traverse the rest of the array and maintain the k largest elements in the heap
    for i in range(k, len(arr)):
        if arr[i] > heap[0]:  # Compare with the root of the heap (smallest element)
            heap[0] = arr[i]  # Replace root with the current element
            heapify(heap, k, 0)  # Restore heap property

    return heap  # Heap contains the k largest elements

""" 
Using heapq util
 """
import heapq
def findkLargest(arr, k):

    n=len(arr)
    min_heap= arr[0:k]
    heapq.heapify(min_heap)

    for ele in arr[k:]:
        if ele>min_heap[0]:
            heapq.heappushpop(min_heap, ele)

    return sorted(min_heap, reverse=True) # Optional: to return in descending order

""" 
build a min-heap from an array
O(n)/O(logn)
This is because the nodes near the leaves (which are most of the nodes) require much less work to heapify 
than the nodes near the root. As a result, the overall time complexity for building a heap is 
O(n) and not O(nlogn).
 """
def build_min_heap(arr):

    # Function to heapify a subtree rooted at index i in an array of size n
    def heapify(arr, n, i):
        smallest = i  # Initialize smallest as root
        left = 2 * i + 1  # Left child index
        right = 2 * i + 2  # Right child index

        # If left child is smaller than root
        if left < n and arr[left] < arr[smallest]:
            smallest = left

        # If right child is smaller than the smallest so far
        if right < n and arr[right] < arr[smallest]:
            smallest = right

        # If smallest is not root, swap and continue heapifying
        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]  # Swap
            heapify(arr, n, smallest)  # Recursively heapify the affected subtree

    n = len(arr)
    
    # Start from the last non-leaf node and move upward, applying heapify
    start_idx = n // 2 - 1
    
    # Perform reverse level order traversal and heapify each node
    for i in range(start_idx, -1, -1):
        heapify(arr, n, i)


""" 
Find median of running number stream
O(nlogn)/O(n)
Using heapq lib
 """
class MedianFinder:
    def __init__(self):
        self.leftMaxHeap=[]
        self.rightMinHeap=[]

    def addNum(self, num):
        if self.leftMaxHeap:
            if num<=heapq.heappop(self.leftMaxHeap):
                heapq.heappush(self.leftMaxHeap, num)
            else:
                heapq.heappush(self.rightMinHeap, num)
        else:
            heapq.heappush(self.leftMaxHeap, num)

        if len(self.leftMaxHeap)> len(self.rightMinHeap)+1:
            temp= heapq.heappop(self.leftMaxHeap)
            heapq.heappush(self.rightMinHeap, temp)
        elif len(self.rightMinHeap)>len(self.leftMaxHeap):
            temp= heapq.heappop(self.rightMinHeap)
            heapq.heappush(self.leftMaxHeap, temp)

    def getMedian(self):
        if len(self.leftMaxHeap)>len(self.rightMinHeap):
            return self.leftMaxHeap[0]
        else:
            return (self.leftMaxHeap[0]+self.rightMinHeap[0])/2


""" 
Sort a nearly sorted (or K sorted) array
 """
""" 
Using heapq library
O(nlogk)/O(k)
 """
import heapq
def sortKsortedArray(arr, k):
    heap= arr[:k+1]
    heapq.heapify(heap)

    remaining_arr= arr[k+1:]
    target_index=0
    for ele in remaining_arr:
        min_ele= heapq.heappushpop(heap, ele)
        arr[target_index]=min_ele
        target_index+=1

    while heap:
        arr[target_index]=heapq.heappop(heap)
        target_index+=1

""" 
Without library
O(nlogk)/O(k)
 """
class MinHeap:
    def __init__(self):
        self.heap = []
    
    def insert(self, value):
        # Add value to the heap
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_min(self):
        if len(self.heap) == 0:
            return None
        # Swap the root with the last element
        root = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self._heapify_down(0)
        return root
    
    def _heapify_up(self, index):
        parent_index = (index - 1) // 2
        if index > 0 and self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)
    
    def _heapify_down(self, index):
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._heapify_down(smallest)
    
    def size(self):
        return len(self.heap)

def sort_k_sorted_array(arr, k):
    result = []
    min_heap = MinHeap()

    # Step 1: Insert first k+1 elements into the min-heap
    for i in range(k + 1):
        min_heap.insert(arr[i])
    
    # Step 2: Process the remaining elements of the array
    for i in range(k + 1, len(arr)):
        result.append(min_heap.extract_min())
        min_heap.insert(arr[i])
    
    # Step 3: Extract the remaining elements from the heap
    while min_heap.size() > 0:
        result.append(min_heap.extract_min())

    return result

""" 
Sort numbers stored on different machines 
Or
Merge k sorted arrays
Or
Print all elements in sorted order from row and column wise sorted matrix

Approach: add first elements of all arrays to min heap, then pop element one after other and add next element from same machine(if available).
Note: tuples support heapify, first element is considered for heapifying
O(nlogk)/O(n) - where n is total elements and k is number of machines
 """
def sortNumbersStoredOnDiffMachines(sortedMachineArrays):
    heap=[]
    result=[]

    for maxchine_index in range(len(sortedMachineArrays)):
        ele_index=0
        heapq.heappush(heap, (sortedMachineArrays[machine_index][ele_index], machine_index, ele_index))

    while heap:
        min_ele, machine_index, ele_index= heapq.heappop(heap)
        result.append(min_ele)
        next_ele_index= ele_index+1
        if next_ele_index< len(sortedMachineArrays[machine_index]):
            next_ele= sortedMachineArrays[machine_index][next_ele_index]
            heapq.heappush(heap, (next_ele, machine_index, next_ele_index))

    return result

""" 
kth smallest element in unsorted array
Using Negating Values for a Max-Heap
O(nlogk)/O(k)
 """
def kthSmallestElement(arr, k):
    max_heap= [-ele for ele in arr[:k]]
    heapq.heapify(max_heap)

    for remaining_ele in arr[k:]:
        max_ele= max_heap[0]
        if remaining_ele<-max_ele:
            heapq.heappushpop(max_heap, -remaining_ele)

    return -max_heap[0]

""" 
kth largest element in stream
O(nlogk)/log(k)
 """
class kthLargestOfStream:
    def __init__(self, initialStream, k) -> None:
        self.stream=initialStream
        self.k=k
        for ele in initialStream:
            self.addEle(ele)

    def addEle(self, ele):
        if len(self.stream)<self.k:
            heapq.heappush(self.stream, ele)
        else:
            if ele>self.stream[0]:
                heapq.heapreplace(self.stream, ele)

    def get_kth_largest(self):
        if len(self.stream)==self.k:
            return self.stream[0]
        else:
            return f'provide minimum {self.k} elements'
        
# kth= kthLargestOfStream([],3)

# kth.addEle(10)
# print(kth.get_kth_largest())
# kth.addEle(20)
# kth.addEle(11)
# print(kth.get_kth_largest())
# kth.addEle(70)
# kth.addEle(50)
# print(kth.get_kth_largest())
# Example usage:
# arr = [7, 10, 4, 3, 20, 15]
# arr = [4, 10, 7, 3, 8, 9, 11, 2, 5]
# build_min_heap(arr)
# print("Min-Heap array:", arr)


# Example usage
# arr = [4, 10, 7, 3, 8, 9, 11, 2, 5]
# k = 4
# result = find_k_largest(arr, k)
# print("The", k, "largest elements are:", result)

# mf = MedianFinder()
# mf.addNum(1)
# mf.addNum(2)
# print(mf.findMedian())  # Output: 1.5
# mf.addNum(3)
# print(mf.findMedian())  # Output: 2

# arr= [6, 5, 3, 2, 8, 10, 9]
# k = 3 
# sortKsortedArray(arr, k)
# print(arr)

