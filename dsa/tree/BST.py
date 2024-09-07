class BSTNode:
    def __init__(self, data) -> None:
        self.data=data
        self.leftChild=None
        self.rightChild=None

newBST= BSTNode(None)

def insert(rootNode: BSTNode, nodeValue):
    if not rootNode.data:
        rootNode.data=nodeValue
    elif nodeValue<=rootNode.data:
        if not rootNode.leftChild:
            rootNode.leftChild= BSTNode(nodeValue)
        else:
            insert(rootNode.leftChild, nodeValue)
    else:
        if not rootNode.rightChild:
            rootNode.rightChild= BSTNode(nodeValue)
        else:
            insert(rootNode.rightChild, nodeValue)
    return 'inserted'

def search(rootNode: BSTNode, nodeValue):
    if not rootNode:
        return 'not found'
    elif nodeValue==rootNode.data:
        return 'found'
    elif nodeValue<rootNode.data:
        return search(rootNode.leftChild, nodeValue)
    else:
        return search(rootNode.rightChild, nodeValue)
        
newBST= BSTNode(3)
left= BSTNode(1)
right= BSTNode(5)
newBST.leftChild= left
newBST.rightChild=right

print(search(newBST, 1))