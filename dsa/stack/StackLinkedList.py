import re


class Node:
    def __init__(self, value):
        self.value= value
        self.next= None

class LinkedList:
    def __init__(self) -> None:
        self.head= None

    def __iter__(self):
        curr= self.head
        while curr:
            yield curr
            curr= curr.next

class Stack:
    def __init__(self) -> None:
        self.ll= LinkedList()

    def __str__(self) -> str:
        values= [str(ele.value) for ele in self.ll]
        return '\n'.join(values)
        
    def isEmpty(self):
        if not self.ll.head:
            return True
        else:
            return False

    def push(self, value):
        node= Node(value)
        node.next= self.ll.head
        self.ll.head= node

    def pop(self):
        if self.isEmpty():
            return "empty"
        else:
            popped= self.ll.head.value
            self.ll.head= self.ll.head.next
            return popped

    def peek(self):
        if self.isEmpty():
            return "empty"
        else:
            return self.ll.head.value

    def delete(self):
        self.ll.head= None

stack= Stack()

stack.push(1)
stack.push(2)
stack.push(3)

print(stack)
