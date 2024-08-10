
class Queue:
    def __init__(self, maxSize) -> None:
        self.list= [None]*maxSize
        self.maxSize=maxSize
        self.start=-1
        self.top=-1

    def __str__(self) -> str:
        if self.top>=self.start:
            print(self.list[self.start:self.top+1])
        if self.top<self.start:
            print(self.list[self.start:self.maxSize])
            print(self.list[0:self.top+1])

    def isFull(self):
        if self.top+1==self.start:
            return True
        elif self.start==0 and self.top+1==self.maxSize:
            return True
        else:
            return False

    def isEmpty(self):
        if self.top==-1:
            return True
        else:
            return False

    def enquue(self, value):
        if self.isFull:
            return "queue is full"
        else:
            if self.top+1==self.maxSize:
                self.top=0
            else:
                self.top=+1
                if self.start==-1:
                    self.start=0
            self.list[self.top]=value
            return "element added"

    def dequeue(self):
        if self.isEmpty():
            return "queue is empty"
        else:
            returnVal= self.list[self.start]

            if self.start<self.top:
                self.start=+1
            elif self.start==self.top:
                self.start= -1
                self.top= -1
            else:
                self.start=0

            self.list[self.start]= None
            return returnVal

    def peek(self):
        if self.isEmpty():
            return "queue is empty"
        else:
            return self.list[self.start]

    def delete(self):
        self.list= [None]*self.maxSize
        self.start=-1
        self.top=-1