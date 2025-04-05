class TreeNode:
    def __init__(self, data) -> None:
        self.data= data
        self.leftChild=None
        self.rightChild=None


""" 
Inorder successor
 """
def inorderSuccessor(root, target):
    if root is None:
        return None
    
    succesor=None
    while root:
        if target<root.data:
            succesor= root
            root= root.leftChild
        else:
            root= root.rightChild
    return succesor

""" 
Inorder predecessor
 """
def inorderPredesessor(root, target):
    if root is None:
        return None
    
    predecessor=None
    while root:
        if target>root.data:
            succesor= root
            root= root.rightChild
        else:
            root= root.leftChild
    return predecessor

""" 
Construct BST from Preorder Traversal
 """
def bst_from_preorder(preorder):
    def helper(bound=float('inf')):
        nonlocal idx
        if idx == len(preorder) or preorder[idx] > bound:
            return None
        root_val = preorder[idx]
        idx += 1
        root = TreeNode(root_val)
        root.leftChild = helper(root_val)
        root.rightChild = helper(bound)
        return root

    idx = 0
    return helper()
