# Ali-ICPR MTWI 2018挑战赛：网络图片文字识别

1. 训练数据集分为image与txt，从txt中读取图片标注框的坐标信息，从对应的图片中截取标注部分，并保存，以文本框内容命名，得到清洗后的训练数据集image_clean；

<<<<<<< HEAD
2. 使用tesseract tool, 安装中文简体语言包，可对图片进行文字检测：
	import os
	import sys
	os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
	
	import pyocr
	from PIL import Image
	
	tools = pyocr.get_available_tools()[:]
	if(0 == len(tools)):
		print('No usable tools')
		exit(1)
	print(tools[0].image_to_string(Image.open('./demos/2.png'), lang='chi_sim'))
	# 求识别OCR第1代

3. 选择模型，使用数据集image_clean训练 ———— 模型待定；

Version：
  OS: win_amd64
=======
2. 选择模型，使用数据集image_clean训练 ———— 模型待定；

Version：
  OS：Windows7
>>>>>>> f1b9bb017b4467e9333d86bcea4dfd49c1e97ced
  python：3.5.2
  numpy: 1.13.1
  tensorflow: 1.4.0
  tesseract: 4.0
  
联系我：hsingmin.lee@yahoo.com

