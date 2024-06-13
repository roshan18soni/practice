from asyncio import create_subprocess_shell
from multiprocessing import dummy
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
Remove duplicates from a sorted linked list
O(n)/O(1)
 """
def removeDuplicatesFromSortedList(head):
    prev=head
    curr=head.next

    while curr:
        nxt= curr.next
        if curr.value==prev.value:
            prev.next=nxt
            curr.next=None
        else:
            prev= curr
        curr= nxt

""" 
Remove duplicates from a unsorted linked list
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
Time: O(n)- While loop takes O(N/K) time and inner for loop takes O(K) time. So N/K * K = N. Therefore TC O(N)
Space: O(1)
Watch youtube vedio 'Take U Forward'
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
        if not kthNode:
            if previousNode:
                previousNode.next=temp
            return head
        else:
            next= kthNode.next
            kthNode.next= None

            reversedHead= reverse(temp)

            if temp==head:
                head=reversedHead
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
Add two numbers represented by Linked List, each element is single digit number
O(n)/O(1)
 """
def addTwoNumbers(head1, head2):
    #assuming the elements in both lists are inserted in reversed fashion
    curr1= head1
    curr2= head2
    new_curr=None
    carry=0
    while curr1 or curr2:
        x= curr1.value if curr1 else 0
        y= curr2.value if curr2 else 0

        total= x + y + carry
        
        digit= total%10
        carry= total//10

        #adding node in reversed fashion
        new_node= Node(digit)
        new_node.next= new_curr
        new_curr=new_node
        
        if curr1:
            curr1= curr1.next
        if curr2:
            curr2= curr2.next

    #handling final carry
    if carry:
        #adding node in reversed fashion
        new_node= Node(carry)
        new_node.next= new_curr
        new_curr=new_node

    return new_curr


l1= LinkedList()

for i in range(1, 4):
    l1.append(i)

l1.head.arbit=l1.head.next.next

head= cloneUsingHashing(l1.head)

printit(head)


