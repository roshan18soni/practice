from locale import currency
from multiprocessing.spawn import prepare
from os import pread
from re import search


class Node:
    def __init__(self, value) -> None:
        self.value= value
        self.next= None
        self.arbit= None

    def __str__(self) -> str:
        return str(self.value)


class LinkedList:
    def __init__(self) -> None:
        self.head= None
        self.tail= None
        self.length= 0

    def append(self, value):
        new_node= Node(value)

        if self.head is None:
            self.head= new_node
            self.tail= new_node
        else:
            self.tail.next= new_node
            self.tail= new_node

        self.length+=1
    
    def prepend(self, value):
        new_node= Node(value)

        if self.head is None:
            self.head= new_node
            self.tail= new_node
        else:
            new_node.next= self.head
            self.head= new_node

        self.length+=1

    def insert(self, index, value):
        new_node= Node(value)

        if 0>index>self.length:
            raise Exception('out of index')
        elif index==0:
            self.prepend(value)
        elif index==self.length:
            self.append(value)
        else:
            temp_node= self.head

            for _ in range(index-1):
                temp_node= temp_node.next

            new_node.next= temp_node.next
            temp_node.next= new_node
            self.length+=1

    def traverse(self):
        current_node= self.head
        while current_node:
            print(current_node.value)
            current_node=current_node.next

    def search(self, value):
        current= self.head

        while current:
            if current.value==value:
                return True
            current=current.next
            
        return False

    def get(self, index):

        if 0>index>=self.length:
            return None

        if index==self.length:
            return self.tail
        else:    
            current= self.head

            for _ in range(index):
                current= current.next

            return current

    def set_value(self, index, value):
        current= self.get(index)
        if current:
            current.value= value
            return True
        return False

    def pop_first(self):
        
        if self.length==0:
            return None

        popped_node= self.head

        if self.length==1:
            self.head=None
            self.tail=None
        else:
            self.head=self.head.next
 
        popped_node.next=None
        self.length-=1
        return popped_node

    def pop(self):
        if self.length==0:
            return None

        popped_node= self.tail

        if self.length==1:
            self.head=None
            self.tail=None
        else:
            current_node=self.head
            while current_node.next!= self.tail:
                current_node= current_node.next

            current_node.next=None
            self.tail=current_node
        self.length-=1

        return popped_node

    def remove(self, index):

        if 0>index>=self.length:
            return None
        elif index==0:
            return self.pop_first()
        elif index==self.length-1:
            return self.pop()
        else:
            prev_node=self.get(index-1)
            popped_node= prev_node.next
            prev_node.next=popped_node.next
            popped_node.next=None
            return popped_node

    def delete_all(self):
        self.head=None
        self.tail=None
        self.length=0

    def reverse(self):

        if self.length>1:
            
            prev_node=None
            current_node=self.head

            while current_node:
                next_node= current_node.next
                current_node.next= prev_node
                prev_node= current_node
                current_node= next_node

            self.head, self.tail = self.tail, self.head

    def middle(self):
        slow= self.head
        fast= self.head

        while fast and fast.next:
            slow= slow.next
            fast= fast.next.next

        return slow

    def remove_duplicates(self):

        if self.length>1:
            seen= set()

            prev= None
            current= self.head

            while current:
                if current.value in seen:
                    to_be_removed= current
                    prev.next= current.next
                    current= current.next
                    to_be_removed.next= None
                    self.length-=1
                else:
                    seen.add(current.value)
                    prev= current
                    current= current.next
            self.tail= prev

    def __str__(self) -> str:
        current_node= self.head
        result= ''
        while current_node:
            result+= str(current_node.value)
            current_node= current_node.next
            if current_node:
                result+= '->'

        return result
    
