class DIS_DataLoader(object):
    def __init__(self):
        self.data = []
        self.ptr = 0


    def add_element(self,data):
        self.data.append(data)

    def __next__(self):
        if(self.ptr<self.__len__()):
            result = self.data[self.ptr]
            self.ptr += 1
            return result
        else:
            self.ptr = 0
            raise StopIteration

    def __len__(self):
        return len(self.data)

    def clear_(self):
        self.data = []
        self.ptr = 0

    def __iter__(self):
        while(True):
            try:
                yield self.__next__()
            except:
                break
