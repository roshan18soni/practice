""" 
Level order traversal
 """
from collections import deque
def level_order_traversal(root):
    if not root:
        return
    
    my_q= deque()
    my_q.appendleft(root)
    while my_q:
        ele= my_q.pop()
        print(ele)
        left=ele.left
        right= ele.right
        if left:
            my_q.appendleft(left)
        if right:
            my_q.appendleft(right)
