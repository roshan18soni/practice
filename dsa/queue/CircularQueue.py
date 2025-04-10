class Queue:
    def __init__(self, maxSize) -> None:
        self.list = [None] * maxSize
        self.maxSize = maxSize
        self.start = -1
        self.top = -1

    def __str__(self) -> str:
        if self.isEmpty():
            return "Queue is empty"
        elif self.top >= self.start:
            return str(self.list[self.start:self.top + 1])
        else:
            return str(self.list[self.start:self.maxSize] + self.list[0:self.top + 1])

    def isFull(self):
        if (self.top + 1) % self.maxSize == self.start:
            return True
        else:
            return False

    def isEmpty(self):
        if self.top == -1:
            return True
        else:
            return False

    def enqueue(self, value):
        if self.isFull():
            return "queue is full"
        else:
            if self.top == -1:
                self.start = 0
                self.top = 0
            else:
                self.top = (self.top + 1) % self.maxSize
            self.list[self.top] = value
            return "element added"

    def dequeue(self):
        if self.isEmpty():
            return "queue is empty"
        else:
            returnVal = self.list[self.start]
            self.list[self.start] = None

            if self.start == self.top:
                self.start = -1
                self.top = -1
            else:
                self.start = (self.start + 1) % self.maxSize

            return returnVal

    def peek(self):
        if self.isEmpty():
            return "queue is empty"
        else:
            return self.list[self.start]

    def delete(self):
        self.list = [None] * self.maxSize
        self.start = -1
        self.top = -1