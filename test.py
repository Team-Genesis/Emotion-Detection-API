import io, requests, cv2, numpy as np

url = "https://images.pexels.com/photos/236047/pexels-photo-236047.jpeg"
img_stream = io.BytesIO(requests.get(url).content)
print(type(img_stream))
img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)

cv2.imshow("img", img)
cv2.waitKey(0)