class TreeNode:
    def __init__(self, data) -> None:
        self.data= data
        self.leftChild=None
        self.rightChild=None

bTree= TreeNode('drinks')
leftChild= TreeNode('hot')
rightChild= TreeNode('cold')

bTree.leftChild=leftChild
bTree.rightChild=rightChild

def preOrderTraversal(rootNode:TreeNode):
    print(rootNode.data)
    if rootNode.leftChild:
        preOrderTraversal(rootNode.leftChild)
    if rootNode.rightChild:
        preOrderTraversal(rootNode.rightChild)

def inOrderTraversal(rootNode:TreeNode):
    if rootNode.leftChild:
        inOrderTraversal(rootNode.leftChild)
    print(rootNode.data)
    if rootNode.rightChild:
        inOrderTraversal(rootNode.rightChild)

def postOrderTraversal(rootNode:TreeNode):
    if rootNode.leftChild:
        inOrderTraversal(rootNode.leftChild)
    if rootNode.rightChild:
        inOrderTraversal(rootNode.rightChild)
    print(rootNode.data)

from collections import deque
def levelOrderTraversal(rootNode:TreeNode):
    myQueue= deque()

    if rootNode:
        myQueue.append(rootNode)
    while myQueue:
        root= myQueue.popleft()
        print(root.data)
        if root.leftChild:
            myQueue.append(root.leftChild)
        if root.rightChild:
            myQueue.append(root.rightChild)

def searchBT(rootNode:TreeNode, nodeValue):
    myQueue= deque()

    if rootNode:
        myQueue.append(rootNode)
    while myQueue:
        root= myQueue.popleft()
        if root.data==nodeValue:
            return True
        if root.leftChild:
            myQueue.append(root.leftChild)
        if root.rightChild:
            myQueue.append(root.rightChild)
    return False

def insertNode(rootNode:TreeNode, newNode):
    myQueue= deque()

    if not rootNode:
        rootNode=newNode
        return 'inserted'
    else:
        myQueue.append(rootNode)

    while myQueue:
        root= myQueue.popleft()
        if root.leftChild:
            myQueue.append(root.leftChild)
        else:
            root.leftChild=newNode
            return 'inserted'
        if root.rightChild:
            myQueue.append(root.rightChild)
        else:
            root.rightChild=newNode
            return 'inserted'
    
#delete node: idea is to replace the node to be deleted with the deepest node and delete the deepest node
def getDeepestNode(rootNode:TreeNode):
    myQueue= deque()

    if rootNode:
        myQueue.append(rootNode)
    while myQueue:
        root= myQueue.popleft()
        if root.leftChild:
            myQueue.append(root.leftChild)
        if root.rightChild:
            myQueue.append(root.rightChild)
    return root

def deleteDeepestNode(rootNode:TreeNode, dNode:TreeNode):
    myQueue= deque()

    if rootNode:
        myQueue.append(rootNode)
    while myQueue:
        root= myQueue.popleft()
        if root==dNode:
            root=None
            return
        if root.leftChild==dNode:
            root.leftChild=None
            return
        else:
            myQueue.append(root.leftChild)

        if root.rightChild==dNode:
            root.rightChild=None
        else:
            myQueue.append(root.rightChild)

def deleteNode(rootNode:TreeNode, node:TreeNode):
    myQueue= deque()

    if rootNode:
        myQueue.append(rootNode)
    while myQueue:
        root= myQueue.popleft()
        if root.data==node.data:
            deepestNode= getDeepestNode(rootNode)
            root.data= deepestNode.data
            deleteDeepestNode(rootNode, deepestNode)
            return 'node deleted'
        if root.leftChild:
            myQueue.append(root.leftChild)
        if root.rightChild:
            myQueue.append(root.rightChild)

    return 'failed to delete'

def deleteBTree(rootNode:TreeNode):
    rootNode.data=None
    rootNode.leftChild=None
    rootNode.rightChild=None

newNode= TreeNode('cofee')
insertNode(bTree, newNode)
levelOrderTraversal(bTree)
print('----')
nodeToDelete= TreeNode('hot')
deleteNode(bTree, nodeToDelete)
levelOrderTraversal(bTree)