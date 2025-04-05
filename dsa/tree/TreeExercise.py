
class TreeNode:
    def __init__(self, data) -> None:
        self.data= data
        self.leftChild=None
        self.rightChild=None
        self.next=None #specific to inorder successor problem
        self.previous=None #specific to converting to doubly ll problem

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
https://www.youtube.com/watch?v=80Zug6D1_r4&t=148s&ab_channel=takeUforward
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

def areIdentical(rootNode1:TreeNode, rootNode2:TreeNode):
    myQueue1= deque()
    myQueue2= deque()
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
""" 

""" 
Using recusion
O(n)/O(n)
 """
def maxDepthOrHeight_rec(rootNode):
    if rootNode is None:
        return 0
    
    left_h= height(rootNode.leftChild)
    right_h= height(rootNode.rightChild)
    
    height= 1+ max(left_h, right_h)
    return height

"""
Using Level order
O(n)/O(w)- here w is max width of any level
 """
def maxDepthOrHeight_levelOrd(rootNode:TreeNode):
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
    inorder_index= inOrderList.index(root)

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
Refer this link for non-recursive approaches, they are also easy: 
https://www.geeksforgeeks.org/print-root-leaf-path-without-using-recursion/
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
""" 

"""
Using Stack
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
Using queue
O(n)/O(w)
 """
def levelOrderTraversalSpiral_queue(root: TreeNode):
    if root is None:
        return
    
    myQueue= deque()
    myQueue.append(root)
    leftToRight=True
    result=[]
    while myQueue:
        node_row=[]
        queueLen= len(myQueue)
        for i in queueLen:
            node=myQueue.popleft()
            if leftToRight:
                index= i
            else:
                index= queueLen-i-1

            node_row[index]=node.data
            if node.leftChild:
                myQueue.append(node.leftChild)
            if node.rightChild:
                myQueue.append(node.rightChild)

            leftToRight=not leftToRight

        result.append[node_row]

    return result

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
def issum(root:TreeNode, sum):
    sum-=root.data

    if root.leftChild is None and root.rightChild is None:
        if sum==0:
            return True
    else:
        if root.left:
            if issum(root.leftChild, sum):
                return True
        elif root.right:
            if issum(root.rightChild, sum):
                return True

    sum+=root.data

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
Populate inorder successor of all nodes
O(n)/O(n)
 """
def populate_inorder_successor(root: TreeNode):
    def inorder_traversal(node):
        nonlocal prevNode
        if root is None:
            return

        inorder_traversal(root.leftChild)
        if prevNode:
            prevNode.next=root
        prevNode=root

        inorder_traversal(root.rightChild)

    prevNode=None
    inorder_traversal(root)

""" 
Return inorder successor of any given node
O(n)/O(n)
 """
def get_inorder_successor(root: TreeNode, givenNodeVal):
    def inorder_traversal(root):
        nonlocal prevNode, successor
        if root is None:
            return

        inorder_traversal(root.leftChild)
        if prevNode and prevNode.data==givenNodeVal:
            successor=root
            return
        prevNode=root

        inorder_traversal(root.rightChild)

    prevNode=None
    successor=None
    inorder_traversal(root)
    print(f'successor of {givenNodeVal} is {successor}')

""" 
Populate inorder predecessor of all nodes
O(n)/O(n)
 """
def populate_inorder_predecessor(root: TreeNode):
    def inorder_traversal(node):
        nonlocal prevNode
        if root is None:
            return

        inorder_traversal(root.leftChild)
        if prevNode:
            root.previous=prevNode
        prevNode=root

        inorder_traversal(root.rightChild)

    prevNode=None
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
    if root in None:
        return
    
    def find_max_sum_util(root: TreeNode, current_sum, current_path, max_sum, max_path):
        if root in None:
            return
        
        current_sum+=root.data
        current_path.append(root.data)

        if root.leftChild in None and root.rightChild is None:
            if current_sum>max_sum:
                max_sum=current_sum
                max_path=current_path[:]

        max_sum, max_path= find_max_sum_util(root.leftChild, current_sum, current_path, max_sum, max_path)
        max_sum, max_path= find_max_sum_util(root.rightChild, current_sum, current_path, max_sum, max_path)

        return max_sum, max_path
    
    max_sum, max_path= find_max_sum_util(root, 0, [], 0, [])
    return max_sum, max_path

""" 
Check whether a given Binary Tree is Complete or not.
A Complete Binary Tree is a binary tree where all levels are fully filled 
except possibly for the last level, which is filled from left to right.
O(n)/O(w)
 """
def is_complete_binary_tree(root: TreeNode):
    if root is None:
        return True

    my_queue= deque()
    my_queue.append(root)
    none_found=False

    while my_queue:
        node= my_queue.popleft()

        if node:
            if none_found:
                return False
            else:
                my_queue.append(node.leftChild)
                my_queue.append(node.rightChild)
        else:
            none_found=True

    return True
    
""" 
iterative approach of pre order traversal
O(n)/O(w)
 """    
def iterative_pre_order_traversal(root: TreeNode):
    if root is None:
        return

    my_stack= []

    my_stack.append(root)

    while my_stack:
        node= my_stack.pop()
        print(node.data)
        if node.rightChild:
            my_stack.append(node.rightChild)
        if node.leftChild:
            my_stack.append(node.leftChild)

""" 
iterative approach of post order traversal
 """
""" 
1- Using two stacks
O(n)/O(n)
 """           
def iterative_postorder_traversal_using_multiple_stacks(root:TreeNode):
    if root is None:
        return

    # Initialize two stacks
    stack1 = []
    stack2 = []

    # Push the root to the first stack
    stack1.append(root)

    # Process all nodes in stack1 and push them onto stack2
    while stack1:
        node = stack1.pop()
        stack2.append(node)

        # Push left and then right children to stack1
        if node.leftChild:
            stack1.append(node.leftChild)
        if node.rightChild:
            stack1.append(node.rightChild)

    # Nodes are stored in reverse postorder in stack2
    # Pop from stack2 to get the correct postorder traversal
    while stack2:
        node = stack2.pop()
        print(node.data, end=" ")

""" 
2- Using single stacks
O(n)/O(w)
 """           
def iterative_postorder_traversal_using_single_stack(root):
    if root is None:
        return

    stack = []
    last_visited = None
    current = root

    while stack or current:
        if current:
            stack.append(current)  # Traverse left subtree
            current = current.leftChild
        else:
            peek_node = stack[-1]
            # Check if right subtree exists and if it's already been visited
            if peek_node.rightChild and last_visited != peek_node.rightChild:
                current = peek_node.rightChild
            else:
                # If right subtree is None or already visited, visit the node
                print(peek_node.data, end=" ")
                last_visited = stack.pop()
        
""" 
Reverse Level Order Traversal of a binary tree
Visiting node from bottom to top and right to left
O(n)/O(n)
 """
def reverse_level_order_traversal(root):
    if root is None:
        return

    queue = []
    stack = []
    
    # Start with the root node in the queue
    queue.append(root)
    
    # Perform level-order traversal (BFS)
    while queue:
        # Dequeue the front node and push it onto the stack
        node = queue.pop(0)
        stack.append(node)
        
        # Enqueue right child first, then left child
        # This ensures that left subtree nodes are processed last and
        # therefore appear at the bottom of the stack.
        if node.rightChild:
            queue.append(node.rightChild)
        if node.leftChild:
            queue.append(node.leftChild)

    # Pop nodes from the stack and print them
    while stack:
        node = stack.pop()
        print(node.data, end=" ")

""" 
Convert Binary Tree to Doubly Linked List using inorder traversal
O(n)/O(h)
 """
def binary_tree_to_doubly_ll(root:TreeNode):
    if root is None:
        return

    prev_node=None
    head=None

    def convert_to_doubly_ll(node:TreeNode):
        if node is None:
            return

        nonlocal prev_node
        nonlocal head

        convert_to_doubly_ll(node.leftChild)
        if prev_node is None:
            head= node
        else:
            prev_node.next= node
            node.previous=prev_node

        prev_node= node
        convert_to_doubly_ll(node.rightChild)

    convert_to_doubly_ll(root)

""" 
Height of tree iteratevely
O(n)/O(w)
 """
def hieght_of_tree_iteratively(root: TreeNode):
    if root is None:
        return

    height=0
    my_queue= deque()
    my_queue.append(root)
    while my_queue:
        height+=1
        level_width= len(my_queue)
        for _ in range(level_width):
            node= my_queue.popleft()
            if node.leftChild:
                my_queue.append(node.leftChild)
            if node.rightChild:
                my_queue.append(node.rightChild)

    return height

""" 
Left view of tree
""" 
"""
Iterative
O(n)/O(w)
 """
def left_view_of_tree(root: TreeNode):
    if root is None:
        return

    left_view_nodes=[]
    my_queue= deque()
    my_queue.append(root)
    while my_queue:
        level_width= len(my_queue)
        for i in range(level_width):
            node= my_queue.popleft()
            if i==0:
                left_view_nodes.append(node.data)
            if node.leftChild:
                my_queue.append(node.leftChild)
            if node.rightChild:
                my_queue.append(node.rightChild)

    return left_view_nodes

"""
Recursive
O(n)/O(n)
 """
def left_view_recursive(root, level=0, max_level=[-1]):
    if root is None:
        return
    
    # If this is the first node of this level, print it
    if level > max_level[0]:
        print(root.data, end=" ")
        max_level[0] = level

    # Recur for left and then right subtree
    left_view_recursive(root.leftChild, level + 1, max_level)
    left_view_recursive(root.rightChild, level + 1, max_level)

""" 
Lowest common ancestor BT
O(n)/O(h)
 """
def find_lca(root, n1, n2):
    # Base case
    if root is None:
        return None

    # If either n1 or n2 matches the root's data, then this node is the LCA
    if root.data == n1 or root.data == n2:
        return root

    # Recursively search for LCA in the left and right subtrees
    left_lca = find_lca(root.leftChild, n1, n2)
    right_lca = find_lca(root.rightChild, n1, n2)

    # If both left and right subtrees contain one of the nodes, then root is LCA
    if left_lca and right_lca:
        return root

    # If only one of the subtrees contains both nodes, return that subtree
    return left_lca if left_lca is not None else right_lca

""" 
Print all nodes at k distance from given node
O(n)/O(n)
https://www.youtube.com/watch?v=i9ORlEy6EsI&t=735s&ab_channel=takeUforward
 """
from collections import defaultdict
def getParentPreorder(root: TreeNode, parent=None, parentDict=defaultdict(None)):
    if root is None:
        return None
    
    parentDict[root]=parent
    parent=root

    getParentPreorder(root.leftChild, parent, parentDict)
    getParentPreorder(root.rightChild, parent, parentDict)


def getNodeByVal(root: TreeNode, val):
    if root is None:
        return None
    
    if root.data==val:
        return root
    
    return getNodeByVal(root.leftChild, val) or getNodeByVal(root.rightChild, val)

def getKDistanceUtil(node: TreeNode, k, parentDict):
    myQueue= deque()
    visited= set()
    i=0

    myQueue.append(node)
    while myQueue and i<k:
        qLen= len(myQueue)
        for _ in range(qLen):
            node= myQueue.popleft()
            visited.add(node)

            if parentDict[node] and parentDict[node] not in visited:
                myQueue.append(parentDict[node])
            if node.leftChild and node.leftChild not in visited:
                myQueue.append(node.leftChild)
            if node.rightChild and node.rightChild not in visited:
                myQueue.append(node.rightChild)

        i+=1

    return myQueue

def getKDistance(root, nodeVal, k):
    parent=None
    parentDict=defaultdict(None)
    getParentPreorder(root, parent, parentDict)
    node= getNodeByVal(root, nodeVal)
    kNodes= getKDistanceUtil(node, k, parentDict)

    while kNodes:
        print(kNodes.popleft().data)


""" 
Right view of tree
O(n)/O(w)
 """
def right_view_of_tree(root: TreeNode):
    if root is None:
        return

    right_view_nodes=[]
    my_queue= deque()
    my_queue.append(root)
    while my_queue:
        level_width= len(my_queue)
        for i in range(level_width):
            node= my_queue.popleft()
            if i==0:
                right_view_nodes.append(node.data)
            if node.rightChild:
                my_queue.append(node.rightChild)    
            if node.leftChild:
                my_queue.append(node.leftChild)
            
    return right_view_nodes

""" 
Print nodes between two given level numbers of a binary tree
O(n)/O(h)
 """
def print_nodes_between_levels(root:TreeNode, low, high):
    if root is None:
        return

    my_queue=deque()
    my_queue.append(root)
    current_level=0

    while my_queue:
        current_level+=1
        if current_level>=low and current_level<=high:
            level_length= len(my_queue)
            for _ in range(level_length):
                node= my_queue.popleft()
                print(node.data)
                if current_level<high:
                    if node.leftChild:
                        my_queue.append(node.leftChild)
                    if node.rightChild:
                        my_queue.append(node.rightChild)
        elif current_level<low:
            level_length= len(my_queue)
            for _ in range(level_length):
                node= my_queue.popleft()
                if node.leftChild:
                    my_queue.append(node.leftChild)
                if node.rightChild:
                    my_queue.append(node.rightChild)

""" 
Find minimum node in BST
Note: inordre traversal of bst gives nodes in ascending order
 """
""" 
recursive
O(n)/O(h)
 """
def find_minimum_node_bst(root: TreeNode):
    if root is None:
        return

    left_min= find_minimum_node_bst(root.leftChild)

    if left_min is None:
        return root
""" 
iterative
O(n)/O(1)
 """
def find_minimum_node_bst_iterative(root:TreeNode):
    current = root

    # Loop to find the leftmost node
    while current.leftChild:
        current = current.leftChild

    return current

""" 
check if binary tree is a BST
O(n)/O(h)
 """
def check_if_bst(root:TreeNode, min_range=float('-inf'), max_range=float('inf')):
    if root is None:
        return True

    if min_range<root.data<max_range:
        return True and check_if_bst(root.leftChild, min_range, root.data) and check_if_bst(root.rightChild, root.data, max_range)
    else:
        return False

""" 
kth smallest node BST O(1) space, Morris inorder traversal
O(n)/O(1)
 """
def kthSmallest(root: TreeNode, k: int) -> int:
    count = 0
    current = root

    while current is not None:
        if current.leftChild is None:
            # Visit this node
            count += 1
            if count == k:
                return current.data
            current = current.rightChild
        else:
            # Find the inorder predecessor of current
            predecessor = current.leftChild
            while predecessor.rightChild is not None and predecessor.rightChild != current:
                predecessor = predecessor.rightChild

            if predecessor.rightChild is None:
                # Make current the right child of its predecessor
                predecessor.rightChild = current
                current = current.leftChild
            else:
                # Revert the changes (restore the tree structure)
                predecessor.rightChild = None
                count += 1
                if count == k:
                    return current.data
                current = current.rightChild

    # If we reach here, then k is larger than the number of nodes in the tree
    return None

""" 
kth largest node BST O(1) space, Reverse Morris inorder traversal
O(n)/O(1)
 """
def kthLargest(root: TreeNode, k):
    count=0
    curr=root
    
    while curr:
        if curr.rightChild is None:
            count+=1
            if count==k:
                return curr.data
            curr=curr.leftChild
        else:
            successor= curr.rightChild
            while successor.leftChild is not None or successor.leftChild!=curr:
                successor=successor.leftChild

            if successor.leftChild is None:
                successor.leftChild=curr
                curr=curr.rightChild
            else:
                count+=1
                successor.leftChild=None
                if count==k:
                    return curr.data
                curr= curr.leftChild

    return None

""" 
Sorted Linked List to Balanced BST
O(n)/O(n)
 """
def balancedBSTfromSortedArray(arr):
    if arr is None:
        return
    
    lenth= len(arr)
    mid=lenth//2

    root=TreeNode(arr[mid])
    left_root= balancedBSTfromSortedArray(arr[:mid])
    right_root= balancedBSTfromSortedArray(arr[mid+1:])

    root.leftChild= left_root
    root.rightChild= right_root

    return root


# root = TreeNode(1)
# root.leftChild = TreeNode(2)
# root.rightChild = TreeNode(3)
# root.leftChild.leftChild = TreeNode(7)
# root.leftChild.rightChild = TreeNode(6)
# root.rightChild.leftChild = TreeNode(5)
# root.rightChild.rightChild = TreeNode(4)

#bst
root = TreeNode(20)
root.leftChild = TreeNode(8)
root.rightChild = TreeNode(22)
root.leftChild.leftChild = TreeNode(4)
root.leftChild.rightChild = TreeNode(12)
root.rightChild.leftChild = TreeNode(21)
root.rightChild.rightChild = TreeNode(23)
getKDistance(root, 22, 2)

