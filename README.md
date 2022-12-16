# Simple_camera_image_recog
Simple Windows 10 camera image recognitions, Windows OS suppport of external devices interfaces via plug and play, serails and communication hubs as standards. The problem is when you study and want some advance features from CV2 but you don't want to have all of riches features and that is third party in some application enrironment. You can do it easy by using Tensorflow and Python, the CV2 using is only for programming interfaces because the camera does not provided and active-X and Direct-X is not target when we using python that included object opsition and image masking I will show you next time that is required for some environments.

## Video capture ##

Video captures can be any interfaces program, remote devices need to be significant by default they need to remarks fields as 0 or 1 or remote devices but if not they determine from channels, telling you that is remote devices and when you work with it work carefully but device is workig the sameway if you setting it correctly.
```
global video_capture_0
video_capture_0 = cv2.VideoCapture(0)
```

## Presentation to Matlib plot ##

Presentation to Matlib plots as continue moving images, from the computer personal camera we inverse channels for familair picture color shades we like but if you see inverse that indicated more detail for computer but not for human us. That is because there are challange between senses of familair and contrst information you notice it instant but you cannot extracting data with the efficents way but working with computer it is all the same.
```
def f1( picture ):
    return tf.constant( picture ).numpy()

def animate( i ):
	ret0, frame0 = video_capture_0.read()
	if (ret0):		
		
		frame0 = tf.image.resize(frame0, [29, 39]).numpy()
		
		temp = img_array = tf.keras.preprocessing.image.img_to_array(frame0[:,:,2:3])
		temp2 = img_array = tf.keras.preprocessing.image.img_to_array(frame0[:,:,1:2])
		temp3 = img_array = tf.keras.preprocessing.image.img_to_array(frame0[:,:,0:1])

		temp = tf.keras.layers.Concatenate(axis=2)([temp, temp2])
		temp = tf.keras.layers.Concatenate(axis=2)([temp, temp3])
		temp = tf.keras.preprocessing.image.array_to_img(
			temp,
			data_format=None,
			scale=True
		)
		temp = f1( temp )
		
		im.set_array( temp )
		result = predict_action( temp )
		print( list_actual_label[result] )
		
	return im,
  
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Execution layer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""  
while True:
	ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
	plt.show()

video_capture_0.release()
cv2.destroyAllWindows()
```

## Model prediction ##
Model prediction map its result to target label
```
list_actual_label = [ 'Shoes', 'Duck' ]

def predict_action ( image ) :
	predictions = model.predict(tf.constant(image, shape=(1, 29, 39, 3) , dtype=tf.float32))
	result = tf.math.argmax(predictions[0])
	return result
  
print( list_actual_label[result] )
```

## Files and Directory ##
```
1. simple_camera.py : result as image
2. supervised_learning.gif : simple codes
3. README.md : readme file
```

## Result image ##
![Alt text](https://github.com/jkaewprateep/Simple_camera_image_recog/blob/main/supervised_learning.gif?raw=true "Title")
