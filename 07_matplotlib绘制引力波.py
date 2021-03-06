# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     07_matplotlib绘制引力波
   Author :       Zeke
   date：          2018/5/27
   Description :
-------------------------------------------------
"""
# 首先下载三个文件
# http://python123.io/dv/H1_Strain.wav
# http://python123.io/dv/L1_Strain.wav
# http://python123.io/dv/wf_template.txt

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

rate_h, hstrain = wavfile.read(r'./data/H1_Strain.wav', 'rb')
rate_l, lstrain = wavfile.read(r'./data/L1_Strain.wav', 'rb')
reftime, ref_H1 = np.genfromtxt(r'./data/wf_template.txt').transpose()

print(rate_h)
print(hstrain)

htime_interval = 1 / rate_h
ltime_interval = 1 / rate_l

htime_len = hstrain.shape[0] / rate_h
htime = np.arange(-htime_len/2, htime_len/2, htime_interval)
ltime_len = lstrain.shape[0] / rate_l
ltime = np.arange(-ltime_len/2, ltime_len/2, ltime_interval)


fig = plt.figure(figsize=(12, 6))

# 绘制H1 Strain
plth = fig.add_subplot(221)
plth.plot(htime, hstrain, 'y')
plth.set_xlabel('Time (seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('H1 Strain')

# 绘制1 Strain
pltl = fig.add_subplot(222)
pltl.plot(ltime, lstrain, 'g')
pltl.set_xlabel('Time (seconds)')
pltl.set_ylabel('L1 Strain')
pltl.set_title('L1 Strain')

#绘制Temple
pltref = fig.add_subplot(212)
pltref.plot(reftime, ref_H1, 'g')
pltref.set_xlabel('Time (seconds)')
pltref.set_ylabel('Template Strain')
pltref.set_title('Template Strain')
fig.tight_layout()

plt.show()








































