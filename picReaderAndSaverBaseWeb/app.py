# import asyncio

# from aiohttp import web

# async def index(request):
#     await asyncio.sleep(0.5)
#     return web.Response(body=b'<h1>Index</h1>', content_type='text/html')

# async def hello(request):
#     await asyncio.sleep(0.5)
#     text = '<h1>hello, %s!</h1>' % request.match_info['name']
#     return web.Response(body=text.encode('utf-8'),content_type='text/html')

# async def init(loop):
#     app = web.Application(loop=loop)
#     app.router.add_route('GET', '/', index)
#     app.router.add_route('GET', '/hello/{name}', hello)
#     srv = await loop.create_server(app.make_handler(), '127.0.0.1', 8000)
#     print('Server started at http://127.0.0.1:8000...')
#     return srv

# loop = asyncio.get_event_loop()
# loop.run_until_complete(init(loop))
# loop.run_forever()

from flask import Flask, request, render_template
import base64
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

def picToBase(fileName):   #根据文件名返回浏览器background-image对应的值可以显示的base64编码字符串
    with open(fileName,'rb') as f:
            base64_data = base64.b64encode(f.read())

    picData = 'url(data:image/jpg;base64,' + base64_data.__str__()[2:-1] + ')'
    return picData

def supportChinese():   #增加支持中文的设定
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

@app.route('/', methods=['GET'])
def home():
    print('send the GET request!')
    return render_template('home.html',data='test',messages='显示图片信息')

@app.route('/', methods=['POST'])
def postHome():
    print('send the POST request!')
    # print(request.form)
    # print(request.files['TheFile'])
    tem = request.files['TheFile']  #得到表单提交的文件
    # f = request.files['TheFile'].read()    #bytes数据
    data = None
    messages = None
    if tem.content_type.startswith('image/'): #如果是图片
        base64_data = base64.b64encode(tem.read())
        data = 'url(data:image/jpg;base64,' + base64_data.__str__()[2:-1] + ')'
        messages = '该图片是'+tem.content_type+'类型，已经显示出来了！'

    elif tem.filename.endswith('.pkl'):  #如果是pkl文件
        supportChinese()
        ax = pickle.load(tem)
        data = None
        with open('tem.jpg','wb') as f:
            plt.savefig(f)
        with open('tem.jpg', 'rb') as f:
            base64_data = base64.b64encode(f.read())
            data = 'url(data:image/jpg;base64,' + base64_data.__str__()[2:-1] + ')'
        messages = '该图片是' + tem.content_type + '类型，已经显示出来了！'
    else:  #其他无法处理情况
        data = 'tem'  # picToBase(fileName)
        messages = '很抱歉，您提交的文件类型暂不支持！'

    return render_template('home.html', data=data, messages=messages)

if __name__ == '__main__':
    app.debug = True
    app.run()