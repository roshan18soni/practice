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
        
        if heap_type=='min':
            smallest= node_index
            left_child_index= node_index*2+1
            right_child_index= node_index*2+2

            if left_child_index< heap.heapSize and heap[left_child_index]<heap[smallest]:
                smallest= left_child_index
            
            if right_child_index< heap.heapSize and heap[right_child_index]<heap[smallest]:
                smallest= right_child_index
            
            if smallest==node_index:
                return
            else:
                heap[node_index], heap[smallest]= heap[smallest], heap[node_index]
                heapifyTree(heap, smallest, heap_type)

        elif heap_type=='max':
            greatest= node_index
            left_child_index= node_index*2+1
            right_child_index= node_index*2+2

            if left_child_index< heap.heapSize and heap[left_child_index]>heap[greatest]:
                greatest= left_child_index
            
            if right_child_index< heap.heapSize and heap[right_child_index]>heap[greatest]:
                greatest= right_child_index
            
            if greatest==node_index:
                return
            else:
                heap[node_index], heap[greatest]= heap[greatest], heap[node_index]
                heapifyTree(heap, greatest, heap_type)

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