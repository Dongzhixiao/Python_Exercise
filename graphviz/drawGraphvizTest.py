

from graphviz import Digraph

def numRangeToColorNum(num,ran,sig):
    tenToHex = int(255*(num-ran[0])/(ran[1]-ran[0]))
    strToReturn = hex(255-tenToHex)
    strToReturn = strToReturn[strToReturn.index('x')+1:]
    if len(strToReturn) == 1:
        strToReturn = '0'+strToReturn
    if sig == 'left':
        strToReturn = '#' + strToReturn + 'ffff'
    elif sig=='mid':
        strToReturn = '#ff' + strToReturn + 'ff'
    elif sig =='right':
        strToReturn = '#ffff' + strToReturn
    return strToReturn

dot = Digraph(comment='The Test Table')

# 添加圆点A,A的标签是Dot A
dot.node('A', '0.5',color='red',style='filled',fillcolor=numRangeToColorNum(0.5,(0,4.5),'mid'))

# 添加圆点 B, B的标签是Dot B
dot.node('B', '2.5',color='green',style='filled',fillcolor=numRangeToColorNum(2.5,(0,4.5),'mid'))
# dot.view()

# 添加圆点 C, C的标签是Dot C
dot.node('C', '3.5',color='blue',style='filled',fillcolor=numRangeToColorNum(3.5,(0,4.5),'mid'))
# dot.view()

# 创建一堆边，即连接AB的两条边，连接AC的一条边。
dot.edges(['AB', 'AC'])
# dot.view()

# 在创建两圆点之间创建一条边
dot.edge('B', 'C', 'b to c')


dot.node('tab',label='''<<TABLE cellspacing="0">
	<TR>
	    <TD BGCOLOR="#ffffff"><FONT>0</FONT></TD>
	    <TD BGCOLOR="#ffe3ff"><FONT>0.5</FONT></TD>
	    <TD BGCOLOR="#ffc7ff"><FONT>1</FONT></TD>
	    <TD BGCOLOR="#ffaaff"><FONT>1.5</FONT></TD>
       <TD BGCOLOR="#ff8eff"><FONT>2</FONT></TD>
        <TD BGCOLOR="#ff72ff"><FONT>2.5</FONT></TD>
        <TD BGCOLOR="#ff55ff"><FONT>3</FONT></TD>
        <TD BGCOLOR="#ff39ff"><FONT>3.5</FONT></TD>
        <TD BGCOLOR="#ff1dff"><FONT>4</FONT></TD>
        <TD BGCOLOR="#ff00ff"><FONT>4.5</FONT></TD>
	</TR>
</TABLE>>''',pos="%100,%1(!)",shape='folder',margin='0')

dot.view()


tem = numRangeToColorNum(0.5,(0,4.5),'left')

theR = [0.5*i for i in range(10)]
theS = []
for i in theR:
    theS.append(numRangeToColorNum(i,(0,4.5),'mid'))
