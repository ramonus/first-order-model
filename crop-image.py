import cv2
from pathlib import Path
import face_alignment
import numpy as np
from PIL import Image
import time

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

def timeit(func):
	def wrapper(*args,**kwargs):
		st = time.time()
		r = func(*args,**kwargs)
		print(f"Time elapsed ({func.__name__}): {time.time()-st:.2f}s")
		return r
	return wrapper

def detect_face(img):
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
	marks = fa.get_landmarks(img)[0]
	minx = min(marks, key=lambda x: x[0])[0]
	maxx = max(marks, key=lambda x: x[0])[0]
	miny = min(marks, key=lambda x: x[1])[1]
	maxy = max(marks, key=lambda x: x[1])[1]
	return (minx, miny, maxx-minx, maxy-miny)

@timeit
def main():
	img = cv2.imread("image.jpg")
	img_raw = img.copy()
	face = (x, y, w, h) = detect_face(img)
	imgface = img[int(y):int(y+h), int(x):int(x+w)]
	gray_face = cv2.cvtColor(imgface, cv2.COLOR_BGR2GRAY)
	eyes = eye_detector.detectMultiScale(gray_face, 1.3, 5)
	for i, (ex, ey, ew, eh) in enumerate(eyes):
		if i == 0:
			eye1 = (ex, ey, ew, eh)
		elif i == 1:
			eye2 = (ex, ey, ew, eh)

		# cv2.rectangle(imgface, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
	if eye1[0] < eye2[0]:
		left_eye = eye1
		right_eye = eye2
	else:
		left_eye = eye2
		right_eye = eye1

	left_eye_center = (
		int(left_eye[0] + (left_eye[2]/2)), int(left_eye[1]+(left_eye[3]/2)))
	left_eye_x, left_eye_y = left_eye_center
	right_eye_center = (
		int(right_eye[0]+(right_eye[2]/2)), int(right_eye[1]+(right_eye[3]/2)))
	right_eye_x, right_eye_y = right_eye_center

	# cv2.circle(imgface, left_eye_center, 2, (255, 0, 0), 2)
	# cv2.circle(imgface, right_eye_center, 2, (255, 0, 0), 2)
	# cv2.line(imgface, right_eye_center, left_eye_center, (67, 67, 67), 2)

	if left_eye_y < right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = 1
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = -1

	# cv2.circle(imgface, point_3rd, 2, (255,0,0), 2)
	# cv2.line(imgface, right_eye_center, left_eye_center, (67,67,67), 2)
	# cv2.line(imgface, left_eye_center, point_3rd, (67,67,67), 2)
	# cv2.line(imgface, right_eye_center, point_3rd, (67,67,67), 2)
	def euclidean_distance(a, b):
		x1, x2 = a
		y1, y2 = b
		return np.sqrt(((x2-x1)*(x2-x1)+((y2-y1)*(y2-y1))))

	a = euclidean_distance(left_eye_center, point_3rd)
	b = euclidean_distance(right_eye_center, left_eye_center)
	c = euclidean_distance(right_eye_center, point_3rd)

	cos_a = (b*b+c*c-a*a)/(2*b*c)
	angle = np.arccos(cos_a)
	angle = (angle*180)/np.pi

	if direction == 1:
		angle = 90-angle

	print("Angle:", angle)
	print(direction)

	new_img = Image.fromarray(img_raw)
	new_img = np.array(new_img.rotate(direction*angle))

	nf = (nx, ny, nw, nh) = detect_face(new_img)

	nx, nw = (nx-nw/2), nw*2
	ny, nh = ny-nw+1.1*nh, nw

	if ny<0:
		nx,nw = nx-ny/2,nw+ny
		ny,nh = 0,nw

	pa = (int(nx), int(ny))
	pb = (int(nx+nw), int(ny+nh))

	nface = new_img[pa[1]:pb[1],pa[0]:pb[0]].copy()

	# cv2.rectangle(new_img, pa, pb, (0, 0, 255), 2)


	# cv2.imshow("", nface)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	cv2.imwrite("image_crop.jpg",nface)


if __name__ == "__main__":
	main()
