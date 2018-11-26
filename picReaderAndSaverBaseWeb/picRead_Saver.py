'''
This is the program to save the original picture of matplotlib module and 
save.
'''
import matplotlib.pyplot as plt
import pickle
def SavePicture(fn):
    """
    A function to save the picture of matplotlib
        :param fn: the name to save the picture
    """
    ax = plt.subplot(111)   
    with open(fn,'wb') as fid:
        pickle.dump(ax, fid)


class PrintAttr(type):
    def __new__(cls,name,bases,attrs):
        print('cls:{}'.format(cls))
        print('name:{}'.format(name))
        print('base:{}'.format(bases))
        print('attrs:{}'.format(attrs))
        return type.__new__(cls,name,bases,attrs)

class A(object,metaclass=PrintAttr):
    id = 1
    name = 'Ab'
    pass
    def __init__(self,id,name):
        self.id = id
        self.name = name
        super(A,self).__init__()
    
# class B(A):
#     pass


if __name__ == "__main__":
    # print('Main function entanss!')
    # print(__doc__)
    # plt.hist([1,2]);
    # SavePicture('C:\\Users\\dell\\Desktop\\aaaaaa.pkl')
    # plt.show()

    with open('C:\\Users\\dell\\Desktop\\aaaaaa.pkl','rb') as f:
        ax = pickle.load(f)
    plt.show()