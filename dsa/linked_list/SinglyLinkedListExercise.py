from SinglyLinkedList import *

def printit(head):
    current= head
    while current:
        print(f'{current.value} -> {current.arbit}' )
        current= current.next

""" 
get Nth node in a Linked List
O(n)/O(1)  
 """
def getNthEelement(head, n):
    current= head
    for _ in range(n):
        current= current.next
    
    return current.value
#or
def getNthEelement2(head, n):
    current = head
    count = 1

    # Traverse the list until the end or until the nth node is reached
    while current is not None:
        if count == n:
            return current.data
        count += 1
        current = current.next

    # Return -1 if the index is out of bounds
    return -1

"""
Given only a pointer/reference to a node to be deleted in a singly linked list
O(n)/O(1)
 """
def deleteNode(node):
    if node==None or node.next==None:
        raise ValueError("Node to be deleted cannot be the last node or None")

    node.value= node.next.value
    node.next= node.next.next

"""
Find Middle of the Linked List
O(n)/O(1)
 """
def middleNode(head):
    slow= head
    fast= head

    while fast and fast.next:
        slow= slow.next
        fast= fast.next.next
    return slow

""" 
Nth node from the end of a Linked List
O(n)/O(1)
 """
def nthNodeFromLast(head, n):
    main_pointer= head
    ref_pointer= head

    for _ in range(n-1):
        if ref_pointer.next:
            ref_pointer=ref_pointer.next
        else:
            raise ValueError('out of range')

    while ref_pointer.next:
        main_pointer= main_pointer.next
        ref_pointer= ref_pointer.next

    return main_pointer

""" 
Delete Linked List
O(n)/O(1)
 """
def deleteLinkedList(head):
    current = head
    while current:
        next_node = current.next  # Store reference to the next node
        current.next = None       # Break the link to the next node
        current = next_node       # Move to the next node
    head = None  # Set head to None to fully delete the list

""" 
Reverse Linked List
O(n)/O(1)
 """
def reverse(head):
    prev= None
    current=head

    while current:
        tmp= current.next
        current.next= prev
        prev= current
        current= tmp

    head= prev
    return head

""" 
Detect loop
Approach: Floy's loop detect, using slow and fast moving pointers
O(n)/O(1)
 """
def detectLoop(head):
    slow= head
    fast= head

    while fast and fast.next:
        slow= slow.next
        fast= fast.next.next

        if slow==fast:
            return True
    return False
    
""" 
Check if palindrome
Approach: reverse second hafl, reverse it and then compare 
O(n)/O(1)
 """
def isPalindrome(head):
    slow= head
    fast= head

    while fast and fast.next:
        slow= slow.next
        fast= fast.next.next

    prev= None
    current= slow
    while current:
        temp= current.next
        current.next= prev
        prev= current
        current= temp

    beginning= head
    end= prev
    
    while beginning and end:
        if beginning.value!=end.value:
            return False

        beginning= beginning.next
        end= end.next

    return True

""" 
Clone a Linked List with next and Random Pointer
Approach: having adjecent cloned nodes
Time: O(n)
Auxiliary Space: O(1)
Total Space: O(n)
 """
def clone(head):

    # Step 1: Create a copy of each node and insert it right next to the original node.
    current= head
    while current:
        nodeCopy= Node(current.value)
        next= current.next
        current.next= nodeCopy
        nodeCopy.next= next
        current= next

    # Step 2: Update the arbit pointers for the newly added nodes.
    current1= head
    while current1:
        if current1.arbit:
            current1.next.arbit= current1.arbit.next
        current1= current1.next.next

    # Step 3: Separate the original and copied nodes to form the cloned list.
    originalHead= head
    clonedHead=head.next

    originalCurrent= originalHead
    clonedCurrent= clonedHead

    while originalCurrent:
        originalCurrent.next= originalCurrent.next.next
        originalCurrent= originalCurrent.next.next
        if originalCurrent:
            clonedCurrent.next= clonedCurrent.next.next
            clonedCurrent= clonedCurrent.next.next

    return clonedHead

""" 
Clone a Linked List with next and Random Pointer using hashing
Approach: Using hashmap to store cloned nodes
Time: O(n)
Auxiliary Space: O(n)
Total Space: O(n)
 """
def cloneUsingHashing(head):
    d= dict()

    original= head
    while original:
        cloned_node= Node(original.value)
        d[original]= cloned_node
        original= original.next

    original= head
    while original:
        cloned_node= d.get(original)
        cloned_node.next= d.get(original.next)
        cloned_node.arbit= d.get(original.arbit)

        original= original.next

    return d.get(head)

""" 
Insert new node into sorted list
Approach: comparing each node
O(n)/O(1)
 """
def insertIntoSortedList(head, new_node):
    if not head:
        return new_node

    if new_node.value< head.value:
        new_node.next= head
        return new_node

    prev= head
    curr= head.next

    while curr and new_node.value > curr.value:
        prev= curr
        curr= curr.next

    prev.next= new_node
    new_node.next= curr

    return head

"""
Intersection of two lists, the end node of one of the linked lists got linked to the second list
""" 
"""
Approach1: using length diff
O(m+n)/O(1)
 """
def getIntersection(head1, head2):

    def getLenght(head):
        curr= head
        length=0
        while curr:
            length+=1
            curr= curr.next
        return length

    length1= getLenght(head1)
    length2= getLenght(head2)

    while length1> length2:
        head1= head1.next
        length1-=1

    while length2> length1:
        head2= head2.next
        length2-=1

    while head1 and head2:
        if head1.value==head2.value:
            return head1
        head1=head1.next
        head2=head2.next

    return None

""" 
Approach2: using swapping pointers
O(m+n)/O(1)
 """
def getIntersection2(head1, head2):
  
    # Maintaining two pointers ptr1 and ptr2 
    # at the heads of the lists
    ptr1 = head1
    ptr2 = head2

    # If any one of the heads is None, there is no intersection
    if not ptr1 or not ptr2:
        return None

    # Traverse through the lists until both pointers meet
    while ptr1 != ptr2:
      
        # Move to the next node in each list and if the one 
        # pointer reaches None, start from the other linked list
        ptr1 = ptr1.next if ptr1 else head2
        ptr2 = ptr2.next if ptr2 else head1

    # Return the intersection node, or None if no intersection
    return ptr1

""" 
Print reverse of a Linked List without actually reversing
""" 

"""
Approach1: using recursion
O(n)/O(n)
 """
def printReverseUsingRecursion(head):

    def printReverse(curr):
        if curr is None:
            return
        
        printReverse(curr.next)
        print(curr.value, end=" ")

"""
Approach1: using stack
O(n)/O(n)
 """
def print_reverse(head):
  
    st = []
    curr = head

    while curr:
        st.append(curr.value)
        curr = curr.next

    while st:
        print(st.pop(), end=" ")

""" 
Remove duplicates from a sorted linked list
""" 
"""
Using Iteration
O(n)/O(1)
 """
def removeDuplicatesFromSortedList(head):
    curr = head

    while curr and curr.next:
        if curr.value == curr.next.value:
            next_next = curr.next.next
            curr.next = next_next
        else:
            curr = curr.next

    return head

""" 
Using recursion
O(n)/O(n)
 """
def removeDuplicatesFromSortedList2(head):
    if head is None:
        return
    if head.next is not None:
        if head.value == head.next.value:
            head.next = head.next.next
            removeDuplicatesFromSortedList2(head)
        else:

            removeDuplicatesFromSortedList2(head.next)


""" 
Remove duplicates from a unsorted linked list
Approach: hashset
O(n)/O(n)
 """
def removeDuplicates(head):

    if head is None or head.next is None:
        return head
        
    seen= set()
    prev=head
    current=head

    while current:
        nxt=current.next
        if current.value in seen:
            prev.next=current.next
            current.next=None
        else:
            seen.add(current.value)
            prev=current

        current=nxt
""" 
Merge two sorted linked lists
Approach: dummy node, whenver a new list is created always use dummy node, 
this makes code cleaner by avoiding extra vars to track head of new list
O(m+n)/O(1)
 """
def mergeSortedLists(head1, head2):
    dummyNode= Node(-1)
    prev= dummyNode
    while head1 and head2:
        if head1.value<=head2.value:
            prev.next= head1
            head1= head1.next
        else:
            prev.next= head2
            head2= head2.next
        prev= prev.next

    prev.next= head1 if head1 else head2

    return dummyNode.next

""" 
Reverse a Linked List in groups of given size
Approach: https://www.youtube.com/watch?v=lIar1skcQYI&ab_channel=takeUforward
Time: O(n)- While loop takes O(N/K) time and inner for loop takes O(K) time. So N/K * K = N. Therefore TC O(N)
Space: O(1)
 """
def reverseKGroup(head, k):
    
    def getKthNode(temp, k):
        current=temp
        while current.next and k>1:
            current=current.next
            k-=1
        
        if k==1:
            return current
        else:
            return None

    temp=head
    previousNode=None

    while temp:
        kthNode= getKthNode(temp, k)
        if kthNode is None:
            if previousNode:
                previousNode.next=temp
            return head
        else:
            next= kthNode.next
            kthNode.next= None

            reverse(temp)

            if temp==head:
                head=kthNode
            else:
                previousNode.next= kthNode

            previousNode= temp
            temp=next
    return head

""" 
Sorted insert for circular linked list
O(n)/O(1)
 """
def insertToSortedCircularSLL(head, value):
    new_node= Node(value)
    #case1: empty CSLL
    if not head:
        new_node.next=new_node
        head= new_node
    #case2: insert before head
    elif value< head.value:
        current= head
        while current.next!=head:
            current=current.next

        new_node.next=head
        current.next=new_node
        head= new_node
    #case3: Insert somewhere after the head
    else:
        current=head
        while current.next!=head and current.next.value<value:
            current=current.next
        new_node.next=current.next
        current.next=new_node
    
    return head    

""" 
Detect and remove the loop
Approach: FLoyd loop finding algo ie slow and fast pointers
O(n)/O(1)
 """
def detectAndRemoveLoop(head):
    if not head or not head.next:
        return
    
    slow = fast = head

    while fast and fast.next:
        slow= slow.next
        fast= fast.next.next

        if slow==fast:
            break

    slow= head

    while slow.next!=fast.next:
        slow=slow.next
        fast=fast.next

    fast.next=None

    return head

""" 
Add two numbers represented by Linked List, each element is single digit number, elements are present in reversed order
O(n)/O(n), space cannot be optimised as its used to store the result
 """
def addTwoNumbers1(head1, head2):
    dummy_node=Node(-1)
    carry=0
    curr1= head1
    curr2= head2
    res_curr=dummy_node

    while curr1 or curr2:
        x= curr1.value if curr1 else 0
        y= curr2.value if curr2 else 0

        sum= x+y+carry
        digit= sum%10
        carry=sum//10

        new_node= Node(digit)
        res_curr.next= new_node
        res_curr=res_curr.next
        if curr1:
            curr1= curr1.next
        if curr2:
            curr2= curr2.next

    if carry:
        new_node= Node(carry)
        res_curr.next=new_node

    return dummy_node.next


l1= LinkedList()

for i in range(1, 4):
    l1.append(i)

l1.head.arbit=l1.head.next.next

head= cloneUsingHashing(l1.head)

printit(head)


