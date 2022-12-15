import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tensorflow as tf

import os
from os.path import exists

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
fig = plt.figure()
image = plt.imread( "F:\\datasets\\downloads\\cats_name\\train\\Symbols\\01.jpg" )
im = plt.imshow( image )

list_actual_label = [ 'Shoes', 'Duck' ]

global video_capture_0
video_capture_0 = cv2.VideoCapture(0)

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
loggings = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\loggings.log"

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

def predict_action ( image ) :
	predictions = model.predict(tf.constant(image, shape=(1, 29, 39, 3) , dtype=tf.float32))
	result = tf.math.argmax(predictions[0])
	return result

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=( 29, 39, 3 )),
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Reshape((234, 32)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(2))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
list_picture = []
list_label = []
path_1 = "F:\\datasets\\downloads\\Duck_Shoe\\Duck\\"
path_2 = "F:\\datasets\\downloads\\Duck_Shoe\\Shoes\\"

for file in os.listdir( path_1 ):
	image = plt.imread( path_1 + file )
	image = tf.image.resize(image, [29, 39]).numpy()

	for i in range ( 40 ) :
		if i % 6 == 0 :
			layer = tf.keras.layers.RandomZoom(.5, .2)
			image = layer( image ).numpy()
			list_picture.append( image )
		elif i % 5 == 0 :
			image = tf.image.random_hue(image, 0.2).numpy()
			image = tf.image.random_flip_up_down(image, 1).numpy()
			list_picture.append( image )
		elif i % 4 == 0 :
			image = tf.image.random_saturation(image, 5, 10, 1).numpy()
			image = tf.image.random_flip_left_right(image, 1).numpy()
			list_picture.append( image )
		elif i % 3 == 0 :
			image = tf.image.random_flip_up_down(image, 1).numpy()
			image = tf.image.random_saturation(image, 5, 10, 1).numpy()
			list_picture.append( image )
		elif i % 2 == 0 :
			image = tf.image.random_flip_left_right(image, 1).numpy()
			image = tf.image.random_hue(image, 0.2).numpy()
			list_picture.append( image )
		else :
			list_picture.append( image )
		
		list_label.append( 1 )

for file in os.listdir( path_2 ):
	image = plt.imread( path_2 + file )
	image = tf.image.resize(image, [29, 39]).numpy()
	
	for i in range ( 40 ) :
		if i % 6 == 0 :
			layer = tf.keras.layers.RandomZoom(.5, .2)
			image = layer( image ).numpy()
			list_picture.append( image )
		elif i % 5 == 0 :
			image = tf.image.random_hue(image, 0.2).numpy()
			image = tf.image.random_flip_up_down(image, 1).numpy()
			list_picture.append( image )
		elif i % 4 == 0 :
			image = tf.image.random_saturation(image, 5, 10, 1).numpy()
			image = tf.image.random_flip_left_right(image, 1).numpy()
			list_picture.append( image )
		elif i % 3 == 0 :
			image = tf.image.random_flip_up_down(image, 1).numpy()
			image = tf.image.random_saturation(image, 5, 10, 1).numpy()
			list_picture.append( image )
		elif i % 2 == 0 :
			image = tf.image.random_flip_left_right(image, 1).numpy()
			image = tf.image.random_hue(image, 0.2).numpy()
			list_picture.append( image )
		else :
			list_picture.append( image )
			
		list_label.append( 0 )

dataset = tf.data.Dataset.from_tensor_slices((tf.constant([list_picture], shape=(len(list_picture), 1, 29, 39, 3), dtype=tf.float32),tf.constant([list_label], shape=(len(list_picture), 1, 1, 1), dtype=tf.int64)))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.Nadam( learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam' )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=2 ,validation_data=(dataset))
model.save_weights(checkpoint_path)

while True:
	ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
	plt.show()

video_capture_0.release()
cv2.destroyAllWindows()


input('...')
