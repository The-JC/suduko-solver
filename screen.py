import numpy as np
import cv2
import helper
import math
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

thresh = 180

def process_sudoku(img):
	# proccessed_img = cv2.cvtColor(np.array(img), cv2.color.COLOR_BGR2GRAY)
	proccessed_img = img
	proccessed_img = cv2.threshold(proccessed_img, thresh, 255, cv2.THRESH_BINARY)[1]
	# proccessed_img = cv2.resize(proccessed_img, (len(img[0])//2,len(img)//2))
	return proccessed_img

def find_edges(img): 
	edges = cv2.Canny(img, 30, 200)
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	img_c = np.zeros(img.shape)
	cv2.drawContours(img_c, contours, -1, (255, 0, 0), 3)
	return img_c

def size(w, h):
	return w*h

def find_sudoku(img):
	bw = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
	# bw = cv2.resize(bw, (len(img[0])//2,len(img)//2))
	edges = cv2.Canny(bw, 100, 200)
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	field_contours = []

	threshold_max_area = 500000
	threshold_min_area = 5000

	threshold_cord = 2
	threshold_size = 2

	for c in contours:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.035 * peri, True)
		(x, y, w, h) = cv2.boundingRect(approx)
		aspect_ratio = w / float(h)

		area = cv2.contourArea(c)

		if area > threshold_min_area and (aspect_ratio >= 0.9 and aspect_ratio <= 1.1):
			similar = False
			for fc, (fx, fy, fw, fh) in field_contours:
				if abs(fx-x) < threshold_cord and abs(fy-y) < threshold_cord:
					if abs(fw-w) < threshold_size and abs(fh-h) < threshold_size:
						similar = True

			if similar:
				continue

			# if area > threshold_max_area:
			# 	cv2.drawContours(color, [c], 0, (0, 0, 255), 3)
			# else:
			# 	cv2.drawContours(color, [c], 0, (0, 255, 0), 3)
			field_contours.append((c, (x, y, w, h)))

	sudoku_fields = []
	for c, (x, y, w, h) in field_contours:
		sudoku_fields.append(helper.Area(x, y, w, h))


	sudoku_field = helper.Area(0,0,0,0)
	for sf in sudoku_fields:
		if sudoku_field == '' or sf.getSize() > sudoku_field.getSize():
			sudoku_field = sf

	sudoku_fields.remove(sudoku_field)
	
	# print('Found ', len(field_contours), ' fields!')
	return (sudoku_field, sudoku_fields)

def crop(img, field, fields=[]):
	cropped = img[field.y:field.y+field.h, field.x:field.x+field.w]
	if fields:
		for f in fields:
			f.x -= field.x
			f.y -= field.y

	return cropped

def resize(factor, img, field='', fields=[]):
	img = cv2.resize(img, (math.floor(len(img[0])*factor), math.floor(len(img)*factor)))

	if fields:
		for f in fields:
			f.x = math.floor(f.x*factor)
			f.y = math.floor(f.y*factor)
			f.w = math.floor(f.w*factor)
			f.h = math.floor(f.h*factor)

	if field != '':
		field.w = math.floor(field.w*factor)
		field.h = math.floor(field.h*factor)

	return img

def square(img, field):
	crop = img[field.y:field.y+field.h, field.x:field.x+field.w]

	arr = np.zeros((3,3))

	for i in range(0, 3):
		y1 = math.floor(field.h/3*i)+4
		y2 = math.floor(field.h/3*(i+1))
		w = field.w
		column = crop[y1:y2, 5:-5]

		data = pytesseract.image_to_boxes(column, output_type="dict", config="--psm 6 -c tessedit_char_whitelist='123456789'")
		if len(data['char']) > 3:
			show(column)
			raise Exception('Found more characters than expected', data.char)
		
		if len(data['char']) > 1 or (len(data['char']) == 1 and data['char'][0] != ''):
			for j in range(0, len(data['char'])):
				x = int(data['left'][j])
				nearest = -1
				for k in range(0, 3):
					if (nearest == -1 or abs(w/3*nearest-x) > abs(w/3*k-x)) and abs(w/3*k-x) < 14:
						nearest = k
				# print("Line ", i,"Num ", data["char"][j], " for ", nearest)
				arr[i, nearest] = data['char'][j]

				# if data['char'][j] == '4':
					# show(column)

		# print(arr[i])
		# show(column)

	# arr = np.zeros((3,3))
	# for i in range(0,9):
	# 	y = math.floor(i/3)
	# 	x = math.floor(i%3)
	# 	y1 = math.floor(field.h/3*y)
	# 	y2 = math.floor(field.h/3*(y+1))
	# 	x1 = math.floor(field.w/3*x)
	# 	x2 = math.floor(field.w/3*(x+1))

	# 	number = crop[y1:y2, x1:x2]
	# 	number = number[4: , 4:]
	# 	number = number[:-4, :-4]
	# 	num = pytesseract.image_to_string(number, config="--psm 10 -c tessedit_char_whitelist='123456789'")
	# 	if num and num != 'a':
	# 		try:
	# 			arr[y,x] = num
	# 		except ValueError:
	# 			show(number)

	# print(arr)
	# show(crop)

	return arr

def remove_noise(image):
    return cv2.medianBlur(image,3)

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def canny(image):
    return cv2.Canny(image, 100, 200)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def show(img):
	cv2.imshow("sudoku", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()