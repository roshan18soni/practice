
class TreeNode:
    def __init__(self, data) -> None:
        self.data= data
        self.leftChild=None
        self.rightChild=None
        self.next=None #specific to inorder successor problem

""" 
Size of a tree
 """
 
""" 
O(n)/O(n)
 """
def sizeOfTree(rootNode:TreeNode):
    if not rootNode:
        return 0
    return 1 + sizeOfTree(rootNode.leftChild) + sizeOfTree(rootNode.rightChild)

""" 
O(n)/O(1)
 """
def sizeOfTreeMorrisInOrderTraversal(rootNode:TreeNode):
    count=0
    current_node= rootNode
    while current_node is not None:
        if current_node.leftChild is None:
            # Case 1: No left child, visit this node and move to the right child
            count+=1
            print(current_node.data)
            current_node= current_node.rightChild
        else:
            # Case 2: Has a left child, find the in-order predecessor
            predecessor=current_node.leftChild

            # Find the rightmost node in the left subtree (predecessor)
            while predecessor.rightChild is not None and predecessor.rightChild!=current_node:
                predecessor= predecessor.rightChild

            # If the right child of the predecessor is None, establish the thread
            if predecessor.rightChild is None:
                predecessor.rightChild=current_node
                current_node= current_node.leftChild
            else:
                predecessor.rightChild=None
                count+=1
                print(current_node.data)
                current_node= current_node.rightChild

""" 
check if two trees are identical
 """

""" 
Using depth first traversal, preorder
O(min(N, M))/O(log min(N, M))
 """
def areIdentical(rootNode1:TreeNode, rootNode2:TreeNode):
    if rootNode1 is None and rootNode2 is None:
        return True
    elif rootNode1 is not None and rootNode2 is not None:
        return rootNode1.data==rootNode2.data and areIdentical(rootNode1.leftChild, rootNode2.leftChild) and areIdentical(rootNode1.rightChild, rootNode2.rightChild)
    else:
        return False

""" 
Using level order traversal
O(N)/O(N)
 """  
from collections import deque
from lib2to3.pytree import Node

def areIdentical(rootNode1:TreeNode, rootNode2:TreeNode):
    myQueue1= deque
    myQueue2= deque
    if rootNode1 is None and rootNode2 is None:
        return True
    elif rootNode1 is not None and rootNode2 is not None:    
        myQueue1.append(rootNode1)
        myQueue2.append(rootNode2)

        while myQueue1 and myQueue2:
            if myQueue1.popleft()==myQueue2.popleft():
                if rootNode1.leftChild and rootNode2.leftChild:
                    myQueue1.append(rootNode1.leftChild)
                    myQueue2.append(rootNode2.leftChild)
                else:
                    return False

                if rootNode1.rightChild and rootNode2.rightChild:
                    myQueue1.append(rootNode1.rightChild)
                    myQueue2.append(rootNode2.rightChild)
                else:
                    return False
            else:
                 return False
    else:
        return False

    return True

""" 
Using Morris inorder traversal
O(n)/O(1)
 """
def morris_in_order_check_identical(root1, root2):
    current1 = root1
    current2 = root2

    while current1 is not None and current2 is not None:
        # If both current nodes are None, trees are identical so far, move to the right subtree
        if current1 is None and current2 is None:
            break
        
        # If one of them is None, the trees are not identical
        if current1 is None or current2 is None:
            return False

        # If left child is None, visit this node and move to right child
        if current1.left is None and current2.left is None:
            if current1.val != current2.val:
                return False
            current1 = current1.right
            current2 = current2.right

        elif current1.left is not None and current2.left is not None:
            # Find the inorder predecessor of current1
            pre1 = current1.left
            while pre1.right is not None and pre1.right is not current1:
                pre1 = pre1.right

            # Find the inorder predecessor of current2
            pre2 = current2.left
            while pre2.right is not None and pre2.right is not current2:
                pre2 = pre2.right

            # Make current1 the right child of its inorder predecessor
            if pre1.right is None and pre2.right is None:
                pre1.right = current1
                pre2.right = current2
                current1 = current1.left
                current2 = current2.left

            # Revert the changes made in the 'if' part to restore the original tree
            # i.e., fix the right child of predecessor
            elif pre1.right == current1 and pre2.right == current2:
                pre1.right = None
                pre2.right = None
                if current1.val != current2.val:
                    return False
                current1 = current1.right
                current2 = current2.right

        else:
            # If one tree has a left child and the other doesn't, they aren't identical
            return False

    # Both current1 and current2 should be None at this point if trees are identical
    return current1 is None and current2 is None

""" 
max depth or height of tree
O(n)/O(w)- here w is max width of any level
 """
def maxDepthOrHeight(rootNode:TreeNode):
    if rootNode is None:
        return 0

    height=0
    myqueue= deque()
    myqueue.append(rootNode)
    while myqueue:
        height+=1
        level_size= len(myqueue)
        for _ in range(level_size):
            node= myqueue.popleft()
            if node.leftChild:
                myqueue.append(node.leftChild)
            if node.rightChild:
                myqueue.append(node.rightChild)
                
    return height

""" 
deleting whole tree, using post order traversal
O(n)/O(n)
 """
def deleteTree(rootNode:TreeNode):
    if rootNode is None:
        print('returning')
        return
    
    deleteTree(rootNode.leftChild)
    deleteTree(rootNode.rightChild)

    print(f'deleting {rootNode.data}')
    rootNode.data=None
    rootNode.leftChild=None
    rootNode.rightChild=None

""" 
mirror a binary tree
 """
""" 
mirror using post order traversal
O(n)/O(n)
 """
def mirrorTreePostOrder(rootNode: TreeNode):
    if rootNode is None:
        return
    
    mirrorTreePostOrder(rootNode.leftChild)
    mirrorTreePostOrder(rootNode.rightChild)

    rootNode.leftChild, rootNode.rightChild= rootNode.rightChild, rootNode.leftChild

    return rootNode
""" 
mirror using level order traversal
O(n)/O(w), w is the max width tree
 """
def mirrorTreeLevelOrder(rootNode: TreeNode):
    if rootNode is None:
        return

    myQueue= deque()
    myQueue.append(rootNode)

    while myQueue:
        root= myQueue.popleft()
        root.leftChild, root.rightChild= root.rightChild, root.leftChild
        if root.leftChild:
            myQueue.append(root.leftChild)
        elif root.rightChild:
            myQueue.append(root.rightChild)

""" 
construct the tree if two type of traversals are given
To construct a binary tree from two types of traversals, you generally need two of the following traversals: 
in-order traversal combined with either pre-order traversal or post-order traversal.
 """
def build_tree_from_inorder_preorder(inOrderList, preOrderList):
    if not inOrderList and not preOrderList:
        return None
    
    root= preOrderList.pop(0)
    inorder_index= inOrderList(root)

    root.left= build_tree_from_inorder_preorder(inOrderList[:inorder_index], preOrderList)
    root.right= build_tree_from_inorder_preorder(inOrderList[inorder_index+1:], preOrderList)

    return root

def build_tree_from_inorder_postorder(inOrderList, postOrderList):
    if not inOrderList and not postOrderList:
        return None
    
    root= postOrderList.pop()
    inorder_index= inOrderList(root)

    root.right= build_tree_from_inorder_postorder(inOrderList[inorder_index+1:], postOrderList)
    root.left= build_tree_from_inorder_postorder(inOrderList[:inorder_index], postOrderList)

    return root

""" 
print root to leaf path, using pre-order traversal
O(n)/O(n)
 """
def printRootToLeafPaths(root: TreeNode, path=[]):
    if not root:
        return None

    path.append(root.data)
    if root.leftChild is None and root.rightChild is None:
        print("->".join(path))
    else:
        printRootToLeafPaths(root.leftChild, path)
        printRootToLeafPaths(root.rightChild, path)

    path.pop()
    
""" 
Lowest Common Ancestor of BST
 """
""" 
 recursive approad
 O(h)/O(h)-  where h is tree hight
"""
def lowestCommonAncestor(root: TreeNode, n1: TreeNode, n2: TreeNode):
    if root is None:
        return
    
    if root>n1 and root>n2:
        return lowestCommonAncestor(root.leftChild, n1, n2)
    elif root<n1 and root<n2:
        return lowestCommonAncestor(root.rightChild, n1, n2)
    else:
        return root

""" 
 iterative approad
 O(h)/O(1)
"""
def lowestCommonAncestor_iter(root: TreeNode, n1: TreeNode, n2: TreeNode):
    if root is None:
        return 

    while root:
        if root>n1 and root>n2:
            root=root.leftChild
        elif root<n1 and root<n2:
            root=root.rightChild
        else:
            return root 
""" 
Count leaf nodes
O(n)/O(n)
 """
def countLeafNodes(root: TreeNode):
    if root is None:
        return 0
    
    if root.leftChild is None and root.rightChild is None:
        return 1
    else:
        return countLeafNodes(root.leftChild) + countLeafNodes(root.rightChild)

""" 
Level order traversal in spiral way
O(n)/O(w): w is width of max nodes of any level
 """
def levelOrderTraversalSpiral(root: TreeNode):
    stack_left_to_right= []
    stack_right_to_left= []

    stack_left_to_right.append(root)

    while stack_left_to_right:
        while stack_left_to_right:
            node= stack_left_to_right.pop()
            print(node.data, end=" ")

            if node.leftChild:
                stack_right_to_left.append(node.leftChild)
            if node.rightChild:
                stack_right_to_left.append(node.rightChild)

        while stack_right_to_left:
            node= stack_right_to_left.pop()
            print(node.data, end=" ")

            if node.rightChild:
                stack_left_to_right.append(node.rightChild)
            if node.leftChild:
                stack_left_to_right.append(node.leftChild)

""" 
diameter of binary tree ie max number of edges between any two nodes.
In the logic, height of one node is considered one, for two nodes its two and so on.. 
hence max sum of left and right subtree gives max number of edges
O(n)/O(n)
 """
def diameter_of_binary_tree(root: TreeNode) -> int:
    def calculate_height_and_diameter(node: TreeNode) -> int:
        nonlocal diameter

        if not node:
            return 0

        # Calculate height of left and right subtrees
        left_height = calculate_height_and_diameter(node.left)
        right_height = calculate_height_and_diameter(node.right)

        # Calculate the diameter passing through this node
        current_diameter = left_height + right_height

        # Update the global diameter
        diameter = max(diameter, current_diameter)

        # Return the height of this node
        return max(left_height, right_height) + 1

    diameter = 0
    calculate_height_and_diameter(root)
    return diameter

""" 
inorder traversal of tree
O(n)/O(h)
alternate approach is morris algo
 """
def inorder_tree_traversal_without_recursion(root: TreeNode):
    if not root:
        return

    my_stack=[]
    current= root
    while current or my_stack:
        while current:
            my_stack.append(current)
            current= current.leftChild

        popped= my_stack.pop()
        print(popped.data)
        current= popped.rightChild

""" 
Root to leaf path sum equal to a given number
O(n/O(h)
 """
def hasPathSum(root: TreeNode, targetSum):
    print(root.data)
    # Base Case 1: If the tree is empty, return False
    if root is None:
        return False
    
    # Base Case 2: If we are at a leaf node, check if targetSum equals the node's value
    if root.leftChild is None and root.rightChild is None:
        return targetSum == root.data
    
    # Recursive Case: Subtract the current node's value from the target sum
    remainingSum = targetSum - root.data
    
    # Check left subtree; if true, stop and return
    if hasPathSum(root.leftChild, remainingSum):
        return True
    
    # Check right subtree only if left subtree didn't return True
    return hasPathSum(root.rightChild, remainingSum)

""" 
Print nodes at k distance from root
O(n)/O(h)
 """
def printKDistant(root: TreeNode, k):
    if root is None:
         return
    
    if k==0:
        print(root.data)
        return

    printKDistant(root.leftChild, k-1)
    printKDistant(root.rightChild, k-1)


""" 
Check if a Binary Tree is subtree of another binary tree
O(n*m)/O(h1*h2)
 """
# Helper function to check if two trees are identical
def areIdentical(root1, root2):
    # Base Case: Both trees are empty
    if root1 is None and root2 is None:
        return True
    
    # If one tree is empty and the other is not
    if root1 is None or root2 is None:
        return False
    
    # Check if the current nodes match and recursively check the left and right subtrees
    return (root1.val == root2.val and
            areIdentical(root1.left, root2.left) and
            areIdentical(root1.right, root2.right))

# Main function to check if S is a subtree of T
def isSubtree(T, S):
    # Base Case: If S is null, it is always a subtree
    if S is None:
        return True
    
    # If T is null and S is not null, then S cannot be a subtree
    if T is None:
        return False
    
    # Check if the trees rooted at T and S are identical
    if areIdentical(T, S):
        return True
    
    # Recursively check the left and right subtrees of T
    return isSubtree(T.left, S) or isSubtree(T.right, S)

""" 
Populate inorder successor
O(n)/O(n)
 """
prevNode=None
def populate_inorder_successor(root: TreeNode):
    def inorder_traversal(node):
        global prevNode
        if root is None:
            return

        populate_inorder_successor(root.leftChild)
        if prevNode:
            prevNode.next=root
            print(f'prevNode.data-{prevNode.data}, prevNode.next.data-{prevNode.next.data}')
        prevNode=root

        populate_inorder_successor(root.rightChild)

    inorder_traversal(root)

"""
Vertical sum in a given binary tree
O(n)/O(n) 
 """
def verticalSum(root:TreeNode):
    if root is None:
            return

    hd_map = {}

    def verticalSumUtil(root:TreeNode, horizontalDistFromRoot=0):
        if root is None:
            return

        verticalSumUtil(root.leftChild, horizontalDistFromRoot-1)

        hd_map[horizontalDistFromRoot]= hd_map.get(horizontalDistFromRoot, 0)+ root.data

        verticalSumUtil(root.rightChild, horizontalDistFromRoot+1)

    verticalSumUtil(root, 0)

    for hd in sorted(hd_map):
        print(f"Vertical sum at horizontal distance {hd}: {hd_map[hd]}")


""" 
Find the maximum sum root to leaf path in a Binary Tree
 """
""" 
using post order traversal
O(n/O(n)
 """ 
def find_max_sum_post_order(root: TreeNode):
    if root is None:
        return 0, []

    def find_max_sum_util(node: TreeNode):
        if node is None:
            return 0, []

        if node.leftChild is None and node.rightChild is None:
            return node.data, [node.data]

        left_sum, left_path= find_max_sum_util(node.leftChild)
        right_sum, right_path= find_max_sum_util(node.rightChild)

        if left_sum>right_sum:
            max_sum= node.data+left_sum
            max_path= left_path+ [node.data]
        else:
            max_sum=node.data+right_sum
            max_path=right_path+[node.data]

        return max_sum, max_path

    max_sum, max_path= find_max_sum_util(root)

    return max_sum, max_path[::-1]

""" 
using pre order traversal
O(n/O(n)
 """ 
def find_max_sum_pre_order(root: TreeNode):
    if root is None:
        return 

    max_sum=[0]
    max_path=[]
    def find_max_sum_util(node: TreeNode, current_sum, current_path):
        if node is None:
            return

        current_sum+=node.data
        current_path.append(node.data)
        if node.leftChild is None and node.rightChild is None:
            if current_sum>max_sum[0]:
                max_sum[0]= current_sum
                max_path.clear()
                max_path.extend(current_path)

        find_max_sum_util(node.leftChild, current_sum, current_path)
        find_max_sum_util(node.rightChild, current_sum, current_path)

        current_path.pop()

    find_max_sum_util(root, 0, [])

    return max_sum[0], max_path

root = TreeNode(1)
root.leftChild = TreeNode(2)
root.rightChild = TreeNode(3)
root.leftChild.leftChild = TreeNode(7)
root.leftChild.rightChild = TreeNode(6)
root.rightChild.leftChild = TreeNode(5)
root.rightChild.rightChild = TreeNode(4)

max_sum, path = find_max_sum_pre_order(root)
print(f"Maximum Sum: {max_sum}")
print(f"Path: {path}")
