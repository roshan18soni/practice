""" 
O(1)/O(n)
 """
class BinaryHeap:
    def __init__(self, size) -> None:
        self.customList= size * [None]
        self.heapSize=0
        self.maxSize=size

""" 
O(1)/O(1)
 """
def peakOfHeap(heap):
    if heap is None:
        return
    
    return heap.customList[0]

""" 
O(1)/O(1)
 """
def sizeOfHeap(heap):
    if heap is None:
        return
    
    return heap.heapSize

""" 
O(n)/O(1)
 """
def levelOrderTraversal(heap):
    if heap is None:
        return
    
    for i in range(0, heap.heapSize):
        print(heap.customList[i])


""" 
O(logN)/O(logN)
 """
def insertNode(heap: BinaryHeap, node, heap_type):

    def heapifyTree(heap: BinaryHeap, index, heap_type):
        if index==0:
            return
        
        parent_index= (index-1)//2

        if heap_type=='min':
            if heap.customList[parent_index]>heap.customList[index]:
                heap.customList[parent_index], heap.customList[index]= heap.customList[index], heap.customList[parent_index]
                heapifyTree(heap, parent_index, heap_type)
        elif heap_type=='max':
            if heap.customList[parent_index]<heap.customList[index]:
                heap.customList[parent_index], heap.customList[index]= heap.customList[index], heap.customList[parent_index]
                heapifyTree(heap, parent_index, heap_type)


    if heap.heapSize==heap.maxSize:
        return "heap is full"

    heap.customList[heap.heapSize]=node
    heapifyTree(heap, heap.heapSize, heap_type)
    heap.heapSize+=1
    return 'node inserted'

""" 
O(logN)/O(logN)
 """
def extractNode(heap: BinaryHeap, heap_type):

    def heapifyTree(heap: BinaryHeap, node_index, heap_type):
        
        left_child_index= 2*node_index+1
        right_child_index= 2*node_index+2

        if left_child_index>=heap.heapSize:
            return
        
        if heap_type=='min':
            smallest_child_index= left_child_index if heap.customList[left_child_index]<heap.customList[right_child_index] else right_child_index
            if heap.customList[node_index]>heap.customList[smallest_child_index]:
                heap.customList[node_index], heap.customList[smallest_child_index]= heap.customList[smallest_child_index], heap.customList[node_index]
                heapifyTree(heap, smallest_child_index, heap_type)
        if heap_type=='max':
            biggest_child_index= left_child_index if heap.customList[left_child_index]>heap.customList[right_child_index] else right_child_index
            if heap.customList[node_index]<heap.customList[biggest_child_index]:
                heap.customList[node_index], heap.customList[biggest_child_index]= heap.customList[biggest_child_index], heap.customList[node_index]
                heapifyTree(heap, biggest_child_index, heap_type)

    if heap.heapSize==0:
        return 'heap is empty'
    
    extractedNode= heap.customList[0]
    heap.customList[0]= heap.customList[heap.heapSize-1]
    heap.customList[heap.heapSize-1]= None
    levelOrderTraversal(heap)
    heap.heapSize-=1
    heapifyTree(heap, 0, 'max')
    
    return extractedNode

""" 
O(1)/O(1)
 """
def deleteEntireBinaryHeap(heap: BinaryHeap):
    heap.customList=None

newHeap= BinaryHeap(5)
insertNode(newHeap, 4, 'max')
insertNode(newHeap, 5, 'max')
insertNode(newHeap, 2, 'max')
insertNode(newHeap, 1, 'max')

levelOrderTraversal(newHeap)

extractNode(newHeap, 'max')

levelOrderTraversal(newHeap)