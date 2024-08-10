class Stack:
    def __init__(self, maxSize) -> None:
        self.list=[]
        self.maxSize= maxSize
    
    def __str__(self) -> str:
        values= [str(v) for v in reversed(self.list)]
        return '\n'.join(values)

    def isEmpty(self):
        if self.list==[]:
            return True
        else:
            return False

    def isFull(self):
        if len(self.list)== self.maxSize:
            return True
        else:
            return False

    def push(self, value):
        if not self.isFull():
            self.list.append(value)
            return "pushed"
        else:
            return "stack full"

    def pop(self):
        if self.isEmpty():
            return 'no element'
        else:
            return self.list.pop()

    def peek(self):
        if self.isEmpty():
            return 'no elements'
        else:
            return self.list[len(self.list)-1]

    def delete(self):
        self.list= None