from ast import Raise



class Queue:
    def __init__(self) -> None:
        self.list= []

    def isEmpty(self):
        if self.list==[]:
            return True
        else:
            return False

    def enqueue(self, value):
        self.list.append(value)
        return "pushed"

    def dequeue(self):
        if not self.isEmpty():
            return self.list.pop(0)
        else:
            raise Exception("empty queue")

    def peek(self):
        if not self.isEmpty():
            return self.list[0]
        else:
            raise Exception("empty queue")

    def delete(self):
        self.list=None