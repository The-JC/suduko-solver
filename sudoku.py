import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import copy

from screen import *
import helper

tick = 0

def recognize(img):
	p_img = process_sudoku(img)
	edges = find_edges(p_img)
	color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	(field, fields) = find_sudoku(img)
	field2 = field
	color = crop(color, field, fields)
	p_img = crop(p_img, field2)

	field.x = field.y = 0

	color = resize(0.5, color, field, fields)
	p_img = resize(0.5, p_img)

	number_array = []

	for f in fields:
		number_array.append(square(p_img, f))
		f.draw(color, width=1)

	number_array.reverse()

	sudoku = np.zeros((9,9))
	for i, block in enumerate(number_array): # Iterate over the 3x3 Blocks
			for j, yc in enumerate(block): # Iterate over the 3x3 cells of a block
				for k, xc in enumerate(yc):
					y = j+i//3*3
					x = k+(i%3)*3
					sudoku[y, x] = xc

	return sudoku

class Sudoku:
	def __init__(self, matrix):
		self.matrix = np.empty((9,9), dtype=object)
		self.original = []
		for i, y in enumerate(matrix):
			for j, x in enumerate(y):
				self.matrix[i, j] = Cell(j, i, x)
				if x != 0:
					self.original.append(self.matrix[i, j])

	def draw(self):
		size = 512
		font = cv2.FONT_HERSHEY_SIMPLEX
		offset_x = 12
		offset_y = -12
		img = np.zeros((size, size, 3))

		for i in range(1, 9):
			thick = 1
			if i%3 == 0:
				thick = 3
			cv2.line(img, (i*size//9, 0), (i*size//9, size), (255, 255, 255), thick)
			cv2.line(img, (0, i*size//9), (size, i*size//9), (255, 255, 255), thick)

		for i in range(0,9):
			for j in range(0,9):
				y = i*size//9
				x = j*size//9
				
				cell = self.matrix[i, j]
				if cell.isAbsolute():
					color = (255, 255, 255)
					if cell in self.original:
						color = (255, 0, 0)
					elif cell in self.changed:
						color = (0, 0, 255)
					cv2.putText(img, str(cell.getValue()), (x+size//(2*9)-offset_x, y+size//(2*9)-offset_y), font, 1, color)
					# cv2.putText(img, str(cell.y) + str(cell.x), (x+size//(2*9)-offset_x, y+size//(2*9)-offset_y), font, 1, (255, 255, 255))
				else:
					for k in range(0, 9):
						if k+1 in cell.getPossibilities():
							dy = (k//3+1)*size//9//3 + offset_y//3
							dx = (k%3)*size//9//3 + offset_x//3

							dy = dy
							dx = dx
							cv2.putText(img, str(k+1), (x+dx, y+dy), font, 0.4, (80, 80, 80))

		cv2.imshow("sudoku", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def isSolved(self):
		for y in self.matrix:
			for x in y:
				if not x.isAbsolute():
					return False

		return True

	def fill_all(self):		
		for i, y in enumerate(self.matrix):
			for j, x in enumerate(y):
				if not x.isAbsolute():
					for k in range(1, 10):
						if not self.isInRow(k, j) and not self.isInColumn(k, y) and not self.isInBlock(k, i, j):
							x.addPossibility(k)

	def setCell(self, y, x, value):
		cell = self.matrix[y, x]
		if cell.isAbsolute():
			return False

		cell.setValue(value)

		# Check row
		for col in self.matrix:
			if not col[x].isAbsolute() and value in col[x].getPossibilities():
				col[x].removePossibility(value)

		# Check column
		for cell in self.matrix[y]:
			if not cell.isAbsolute() and value in cell.getPossibilities():
				cell.removePossibility(value)

		by = y//3*3
		bx = x//3*3

		for i in range(0, 3):
			for j in range(0, 3):
				cell = self.matrix[by+i, bx+j]
				if not cell.isAbsolute() and value in cell.getPossibilities():
					cell.removePossibility(value)

		return

	def checkCellPossiblitiesExclusion(self, cell):
		if cell.isAbsolute():
			return False
		
		for p in cell.getPossibilities():
			if not self.isPossiblityInRow(p, cell.getCol()) or not self.isPossiblityInColumn(p, cell.getRow()) or not self.isPossiblityInBlock(p, cell.getRow(), cell.getCol(), exclude=cell):
				cell.setValue(p)
				self.changed.append(cell)
				return True

		return False

	def checkPossiblitiesExclusion(self):
		changed = False
		for y in self.matrix:
			for c in y:
				if self.checkCellPossiblitiesExclusion(c):
					changed = True

		return changed

	def isPossiblityInBlock(self, num, row, col, exclude=None):
		"""Checks if a possibility is in a block of 3x3
		@param num: possibility to check
		@param row: row to check
		@param col: column to check
		@param exclude: optional cell to ignore
		@type num: int
		@type row: int
		@type col: int
		@type exclude: Cell

		@return: Bool
		"""
		if type(col) is not int or type(row) is not int:
			return
		
		by = row//3*3
		bx = col//3*3

		found = False
		for i in range(0, 3):
			for j in range(0, 3):
				cell = self.matrix[by+i, bx+j]
				if cell != exclude and not cell.isAbsolute() and num in cell.getPossibilities():
					found = True

		return found
	def isPossiblityInRow(self, num, col, exclude=None):
		"""Checks if possiblity is in row
		@param num: possible number to check
		@param col: column index of row to check
		@param exclude: optional cell to ignore
		@type num: int
		@type col: int
		@type exclude: Cell

		@return: Bool
		"""
		if type(num) is not int or type(col) is not int or (exclude and type(exclude) is not Cell):
			raise TypeError('Argument not from expected type!')

		found = False
		for column in self.matrix:
			cell = column[col]
			if cell != exclude and not cell.isAbsolute() and num in cell.getPossibilities():
				found = True
		return found
	def isPossiblityInColumn(self, num, row, exclude=None):
		"""Checks if possiblity is in column
		@param num: possible number to check
		@param row: row index of column to check
		@param exclude: optional cell to ignore
		@type num: int
		@type row: int or [Cell]
		@type exclude: Cell

		@return: Bool
		"""
		if type(num) is not int or (type(row) is not int and type(row) is not np.ndarray) or (exclude and type(exclude) is not Cell):
			raise TypeError('Argument not from expected type!')

		if type(row) is int:
			col = self.matrix[row]
		if type(row) is np.ndarray:
			col = row

		found = False
		for c in col:
			if c != exclude and not c.isAbsolute() and num in c.getPossibilities():
				found = True
				
		return found

	def checkCellNumExclusion(self, cell):
		if cell.isAbsolute():
			return False

		for p in cell.getPossibilities():
			if self.isInRow(p, cell.getCol()) or self.isInColumn(p, cell.getRow()) or self.isInBlock(p, cell.getRow(), cell.getCol()):
				cell.removePossibility(p)
		
		if len(cell.getPossibilities()) == 1:
			cell.setValue(cell.getPossibilities()[0])
			self.changed.append(cell)
			return True

		return False

	def checkNumExclusion(self):
		"""Checks if only one number can fit in a row, column or block
		@return: bool
		"""
		changed = False
		for y in self.matrix:
			for c in y:
				if self.checkCellNumExclusion(c):
					changed = True

		return changed

	def isCellInBlock(self, cell, num=None):
		"""Checks if number is in block of 3x3
		@param cell: cell to check block
		@param num: optional number to check if None value of cell is used
		@type cell: Cell
		@type num: int

		@return: Bool
		"""
		if type(cell) is not Cell or (num and type(num) is not int):
			raise TypeError('Argument not from expected type!')
		return self.isInBlock(cell.getValue() if not num else num, cell.getRow(), cell.getCol(), exclude=cell)
	def isInBlock(self, num, row, col, exclude=None):
		"""Checks if a number is in a block of 3x3
		@param num: number to check
		@param row: row to check
		@param col: column to check
		@param exclude: optional cell to ignore
		@type num: int
		@type row: int
		@type col: int
		@type exclude: Cell

		@return: Bool
		"""
		if type(num) is not int or type(col) is not int or type(row) is not int or (exclude and type(exclude) is not Cell):
			raise TypeError('Argument not from expected type!')
		
		by = row//3*3
		bx = col//3*3

		found = False
		for i in range(0, 3):
			for j in range(0, 3):
				cell = self.matrix[by+i, bx+j]
				if cell != exclude and cell.getValue() == num:
					found = True

		return found
	def isCellInRow(self, cell, num=None):
		"""Checks if number is in row
		@param cell: cell to check row
		@param num: optional number to check if None value of cell is used
		@type cell: Cell
		@type num: int

		@return: Bool
		"""
		if type(cell) is not Cell or (num and type(num) is not int):
			raise TypeError('Argument not from expected type!')
		return self.isInRow(cell.getValue() if not num else num, cell.getCol(), exclude=cell)
	def isInRow(self, num, col, exclude=None):
		"""Checks if number is in row
		@param num: number to check
		@param col: column index of row to check
		@param exclude: optional cell to ignore
		@type num: int
		@type col: int
		@type exclude: Cell

		@return: Bool
		"""
		if type(num) is not int or type(col) is not int or (exclude and type(exclude) is not Cell):
			raise TypeError('Argument not from expected type!')

		found = False
		for column in self.matrix:
			if column[col] != exclude and column[col].getValue() == num:
				found = True
		return found
	def isCellInColumn(self, cell, num=None):
		"""Checks if number is in column
		@param cell: cell to check column
		@param num: optional number to check if None value of cell is used
		@type cell: Cell
		@type num: int

		@return: Bool
		"""
		if type(cell) is not Cell or (num and type(num) is not int):
			raise TypeError('Argument not from expected type!')
		return self.isInColumn(cell.getValue() if not num else num, cell.getRow(), exclude=cell)
	def isInColumn(self, num, row, exclude=None):
		"""Checks if number is in column
		@param num: number to check
		@param row: row index of column to check
		@param exclude: optional cell to ignore
		@type num: int
		@type row: int or [Cell]
		@type exclude: Cell

		@return: Bool
		"""
		if type(num) is not int or (type(row) is not int and type(row) is not np.ndarray) or (exclude and type(exclude) is not Cell):
			raise TypeError('Argument not from expected type!')

		if type(row) is int:
			col = self.matrix[row]
		if type(row) is np.ndarray:
			col = row

		found = False
		for cell in col:
			if cell != exclude and cell.getValue() == num:
				found = True
		return found

	def suppositionOne(self):
		done = False
		
		while not done:
			print("Trying to solve sudoku using supposition...")
			for y in self.matrix:
				for cell in y:
					if cell.isAbsolute():
						continue
					
					for p in cell.getPossibilities():
						print('-> Trying setting %i to (%i, %i)...' % (p, cell.y, cell.x), end='\r')
						sudoku_copy = copy.deepcopy(self)

						sudoku_copy.setCell(cell.y, cell.x, p)
						if not sudoku_copy.solve(output=False) and not sudoku_copy.sanityCheck():
							cell.removePossibility(p)
						
						if len(cell.getPossibilities()) == 1:
							done = True
							break
					if done:
						break
				if done:
					break
			print('\n\nOne level supposition done!')

		return True

	def solve(self, supposition=0, output=True):
		ticks=0
		change = True
		while not self.isSolved() and change:
			ticks+=1
			self.changed = []
			mode = 'NumExcl'
			tick = ticks
			change = self.checkNumExclusion()
			if not change:
				mode = 'PosExcl'
				change = self.checkPossiblitiesExclusion()
			if not change and supposition == 1:
				mode = 'Sup'
				change = self.suppositionOne()
			if not change:
				mode = 'none'

			if output:
				print("%i -> %s - %s" % (ticks, mode, 'Passed' if self.sanityCheck() else 'Failed'), end='\r')

		if self.isSolved() and self.sanityCheck() and output:
			print("\n\nTook ", ticks, " rounds to solve!\n")
		elif output: 
			print("Couldn't solve in %i\n" % ticks)
			
		return self.isSolved()

	def sanityCheck(self):
		for y in self.matrix:
			for cell in y:
				if not cell.isAbsolute():
					continue
				
				if self.isCellInRow(cell=cell) or self.isCellInColumn(cell=cell) or self.isCellInBlock(cell=cell):
					return False

		return True

	def print(self):
		for y in range(0,9):
			s = ''
			for x in range(0, 9):
				s += str(self.matrix[y, x].toString()) + ' '
			print(s)

	def __str__(self):
		s = ''
		for y in range(0,9):
			for x in range(0, 9):
				s+= str(self.matrix[y, x].toString()) + ' '
			s+= '\n'
		return s

class Cell:
	def __init__(self,  x, y, value=0):
		self.x = x
		self.y = y
		self.value = int(value)
		if self.value == 0:
			self.possibilities = []

	def getRow(self):
		return self.y
	def getCol(self):
		return self.x

	def getValue(self):
		return self.value

	def setValue(self, val, reason=''):
		if val != 0:
			self.possibilities = []
		self.value = val
	
	def getPossibilities(self):
		return self.possibilities

	def addPossibility(self, pos):
		self.possibilities.append(pos)
	
	def removePossibility(self, pos):
		self.possibilities.remove(pos)

	def isAbsolute(self):
		return self.value != 0

	def toString(self):
		return self.getValue() if self.isAbsolute() else 'X'

if __name__ == '__main__':
	print("Sudoku solver v1.0.0\nReading image...")
	img = cv2.imread("e1.PNG", cv2.IMREAD_GRAYSCALE)

	start = time.time()

	print("Analyzing sudoku...")
	sudoku = recognize(img)
	print("Took ", time.time()-start, 's to recognice')

	# sudoku = [[9., 1., 0., 0., 0., 7., 3., 0., 0.],
    #    [0., 0., 0., 0., 9., 0., 5., 0., 0.],
    #    [6., 0., 0., 0., 0., 4., 0., 0., 7.],
    #    [8., 7., 0., 4., 0., 0., 0., 0., 0.],
    #    [2., 0., 6., 0., 8., 0., 0., 0., 3.],
    #    [0., 0., 9., 0., 0., 0., 8., 0., 0.],
    #    [0., 3., 0., 0., 0., 0., 0., 8., 5.],
    #    [0., 0., 0., 6., 0., 0., 0., 0., 0.],
    #    [0., 0., 4., 0., 0., 0., 2., 1., 0.]]

	# sudoku =[[0, 6, 8, 0, 0, 0, 9, 3, 0],[0, 4, 2, 0, 0, 0, 6, 0, 0],[1, 9, 0, 0, 8, 0, 0, 4, 0],[0, 8, 5, 2, 0, 1, 0, 0, 7],[7, 0, 0, 8, 9, 0, 0, 0, 0],[2, 0, 9, 0, 0, 7, 5, 0, 3],[0, 2, 0, 1, 0, 0, 0, 5, 0],[8, 5, 0, 0, 4, 0, 7, 6, 0],[4, 7, 3, 0, 5, 2, 0, 0, 9]]

	sud = Sudoku(sudoku)
	sud.fill_all()
	
	sud.print()
	sud.solve(supposition=1)

	# sud.print()

	sud.draw()

	# cv2.imshow("sudoku", color)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
