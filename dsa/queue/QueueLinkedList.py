class Node:
    def __init__(self, value) -> None:
        self.value= value
        self.next= None

class LinkedList:
    def __init__(self) -> None:
        self.head= None
        self.tail= None

    def __iter__(self):
        curr= self.head
        while curr:
            yield curr
            curr= curr.next

class Queue:
    def __init__(self) -> None:
        self.ll= LinkedList()

    def isEmpty(self):
        if self.ll.head==None:
            return True
        else:
            return False

    def enqueue(self, value):
        node= Node(value)
    
        if self.isEmpty():
            self.ll.head=node
        else:
            self.ll.tail.next=node  
            
        self.ll.tail=node

    def dequue(self):
        if self.isEmpty():
            return "empty queue"
        else:
            node= self.ll.head
            if self.ll.head==self.ll.tail:
                self.ll.tail=None
            self.ll.head=node.next
        
        return node

    def peek(self):
        if self.isEmpty():
            return "empty queue"
        else:
            return self.ll.head

    def delete(self):
        self.ll.head=self.ll.tail=None