# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     05_matplotlib入门
   Author :       Zeke
   date：          2018/5/26
   Description :
-------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
#
# plt.plot(x, y, format_string, **kwargs)
# format_string: 控制曲线的格式字符串，可选由颜色字符/风格字符/标记字符组成
    # 颜色字符
        # 'b'蓝色, 'g'绿色, 'r'红色, 'c'青绿色, 'm'洋红色, 'y'黄色, 'k'黑色, 'w'白色
    # 风格字符
        # '-'实线/  '--'破折线/  '-.'点画线/  ’:‘虚线/  ""无线条
    # 标记字符
        # '.'点标记, ','像素标记(极小点), 'o'实心圈标记, 'v'倒三角标记, '^'上三角标记, '>'右三角标记, '<'左三角标记
        # '1'下花三角, '2'上花三角, '3'左花三角, '4'右花三角, 's'实心方形, 'p'实心五角, '*'星形标记
        # 'h'竖六边形, 'H'横六边形, '+'十字形, 'x'x标记, 'D'菱形, 'd'瘦菱形, '|'垂直线



# 第一张图
plt.plot([0, 2, 4, 6, 8], [3, 1, 4, 5, 2])
plt.ylabel("Grade")
plt.axis([0, 10, 0, 6])
# plt.savefig("test", dpi=600)
plt.show() # 显示图出来


# 第二张图
# plt.subplot(nrows, ncols, plot_number)
# plt.subplot(3, 2, 4)  # 把画图区域分成3行2列，图在第4个区域
# plt.subplot(324)
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

a = np.arange(0, 5, 0.02) # 0到5 步长0.02
# 子图1
plt.subplot(211)
plt.plot(a, f(a))
# 子图2
plt.subplot(2, 1, 2)
plt.plot(a, np.cos(2*np.pi*2*a), 'r--') # r红色，-实线
plt.show()


# 第三张图 一个图里画4种线。
a = np.arange(10)
plt.plot(a,a*1.5,'go-', a,a*2.5,'rx', a,a*3.5,'*', a,a*4.5,'b-.')
plt.show()




# pyplot的中文显示

# 方法1: rcParams修改字体
import matplotlib.pyplot as plt
import matplotlib

# 'font.family' 用于显示字体的名字
    # 'SimHei'黑体， 'Kaiti'楷体, 'LiSu'隶书, 'FangSong'仿宋, 'YouYuan'幼圆, 'STSong'华文宋体
# 'font.style' 字体风格，正常'normal'或斜体'italic'
# 'font.size' 字体大小，整数字号或者'large', 'x-small'

matplotlib.rcParams['font.family'] = 'Kaiti' #黑体
matplotlib.rcParams['font.size'] = '10' #包括坐标轴字号均改变
b = np.arange(1, 5, 0.02)
plt.plot(b, np.cos(2*np.pi*b), 'r--')
plt.xlabel('横轴：时间') # 设定X Y轴的标签
plt.ylabel('纵轴：振幅')
plt.show()


# 方法2: 增加一个属性: fontproperties
plt.rcParams['axes.unicode_minus']=False #解决坐标轴负值不显示问题
b = np.arange(1, 5, 0.02)
plt.plot(b, np.cos(2*np.pi*b), 'r--')
plt.xlabel('横轴：时间',fontproperties='SimHei',fontsize=20)
plt.ylabel('纵轴：振幅',fontproperties='SimHei',fontsize=20)
plt.show()



# pyplot的文本显示
c = np.arange(0, 5, 0.02)
plt.plot(c, np.cos(2*np.pi*c),'r--')

plt.xlabel('横轴：时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴：振幅', fontproperties='SimHei', fontsize=15)
plt.title(r'正弦波实例 $y=cos(2\pi x)$', fontproperties='SimHei',fontsize=25) #$引用Latex语法
plt.text(2, 1, r'$\mu=100$', fontsize=15) #文本显示在横轴为2，纵轴为1的位置上面

plt.axis([-1, 6, -2, 2])
plt.grid(True)
plt.show()

# pyplot绘制带箭头指示的文本
d = np.arange(0, 5, 0.02)
plt.plot(d, np.cos(2*np.pi*d),'r--')

plt.xlabel('横轴：时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴：振幅', fontproperties='SimHei', fontsize=15)
plt.title(r'正弦波实例 $y=cos(2\pi x)$', fontproperties='SimHei',fontsize=25)
plt.annotate(r'$\mu=100$', xy=(2,1), xytext=(3,1.5),
             arrowprops=dict(facecolor='black', shrink=0.1, width=2))
plt.axis([-1,6,-2,2])
plt.grid(True)  # 网格线
plt.show()


# pyplot网格绘图
# plt.subplot2grid(GridSpec, CurSpec, colspan=1, rowspan=1)
# 编号从0开始

# subplot配合gridspec使用
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3,3)

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :-1])
ax3 = plt.subplot(gs[1:, -1])
ax4 = plt.subplot(gs[2, 0])
ax5 = plt.subplot(gs[2, 1])
plt.show()
