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
    # picPath = 'F:\\DoctorContent\\log-related\\aLogAnalysisPrograme\\behaviorialAnalysis\\client51分析结果\\1.pkl'
    # with open(picPath,'rb') as f:
    #     ax = pickle.load(f)
    # plt.show()
    # import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.plot((1, 2, 3), (4, 5, 7))
    plt.xlabel('横坐标')
    plt.ylabel('纵坐标')
    plt.show()
