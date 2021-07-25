from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())


# resimleri yükler, gri tonlamaya dönüştürür ve ardından 7 x 7 çekirdekli bir Gauss filtresi kullanarak bulanıklaştırır .

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)

#Görüntümüz bulanıklaştıktan sonra, görüntüdeki kenarları tespit etmek için Canny kenar dedektörünü uygularız -
#daha sonra kenar haritasındaki herhangi bir boşluğu kapatmak için bir genişletme + erozyon gerçekleştirilir

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#Kenar haritasındaki nesnelerin ana hatlarını algılar.


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#konturlarımızı soldan sağa doğru sıralar. 
# konturları soldan sağa sıralamak, referans nesneye karşılık gelen konturun her zaman ilk girdi olmasını sağlar.

(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 25, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255))
refObj = None


for c in cnts:
	#Kontur yeterince büyük değilse onu göz ardı ederiz.
	if cv2.contourArea(c) < 100:
		continue
	#referans nesnenin döndürülmüş sınırlayıcı kutusunu hesaplar.
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# ardından döndürülen sınırlamanın ana hatlarını çizer
	box = perspective.order_points(box)
	
	# sınırlayıcı kutunun merkezini hesaplar
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])
	#noktalar arasındaki Öklid mesafesini hesaplıyoruz bize "metrik başına piksel sayımızı" vererek, 
	#kaç pikselin sığacağını belirlememizi sağlıyor.
	if refObj is None:
		
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		
		D = (dist.euclidean((tlblX, tlblY), (trbrX, trbrY)))
		refObj = (box, (cX, cY), (D/ args["width"] ))
		continue
	
	# resmin konturlarını çizdirir.
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
	
	#refCoords ve objCoords  sınırlayıcı kutu koordinatları ve  ağırlık merkezinin (x, y) ve  koordinatları aynı dizilere dahil eder.
	refCoords = np.vstack([refObj[0], refObj[1]])
	objCoords = np.vstack([box, (cX, cY)])
	

	for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
		#koordinatlarını temsil eden bir daire çizdiririz  ve noktaları birleştirmek için bir çizgi çizeriz. 
		cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
		cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
		cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
			color, 2)
		#referans konum ile nesne konumu arasındaki Öklid mesafesini hesaplar ve ardından mesafeyi 
		#"metrik piksel başına" bölerek bize iki nesne arasındaki cm cinsinden son mesafeyi verir .
		#Hesaplanan mesafe daha sonra resmimize çizilir. 
		D = round((dist.euclidean((xA, yA), (xB, yB)) / refObj[2]),5)
		(mX, mY) = midpoint((xA, yA), (xB, yB))
		cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
		
		cv2.imshow("Image", orig)
		cv2.waitKey(0)
		a=((int(xA), int(yA)), (int(xB), int(yB)))
		print(((a[0][0]/(10*args["width"]/2.54))),(a[0][1]/(10*args["width"]/2.54)),(a[1][0]/(10*args["width"]/2.54)),(a[1][1]/(10*args["width"]/2.54)))
