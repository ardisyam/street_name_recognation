#%% 
# First Step ...
print("## Start including libraries ...")
print('1. Include our own libraries')
from sources.constant import *
from sources.snr import *
print('2. Include third party libraries')
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
print("## Finish including libraries ...\n")

#%% 
# Second Step ...
print("## Start pre-processing the data ...")
print('3. Get XML files and their path')
xmlfile_list = CreateListOfXmlfile(TRAIN_DIR)
print('4. Get image files and their path')
image_list = CreateListOfImages(xmlfile_list)
print('5. Extract the coordinates of bounding boxes out of the corresponding XML files')
bbox_list = CreateListOfBBox(xmlfile_list)
print('6. Normalize features and targets')
image_features, bb_targets = NormalizeImage(image_list, bbox_list)
print("## Finish pre-processing the data ...\n")


#%%
# Third Step ...
print("## Prepare the training and testing dataset ...")
print('7. Convert features and targets to arrays of float32')
X = np.array(image_features, dtype= np.float32)
y = np.array(bb_targets, dtype= np.float32)
print('8. Split the dataset into training and testing set')
x_train,x_test,y_train,y_test = train_test_split(X,
                                                 y, 
                                                 train_size=0.9,
                                                 random_state=0)
print(f'   The shape of x_train : {x_train.shape}')
print(f'   The shape of x_test  : {x_test.shape}')
print(f'   The shape of y_train : {y_train.shape}')
print(f'   The shape of y_test  : {y_test.shape}')
print("## Finish splitting the training data ...")

# 3. Use a pre-existing model to create a new model. A new model is created
#    by adding more layers. Loss, optimizer, and accuracy metrics are also
#    added in this step.

# %%
# Fourth Step
print("## Start creating the Fully-Connected network ...")
print('9. Get a pre-trained CNN model')
pre_trained_model = InceptionResNetV2(weights='imagenet', 
                                      include_top=False,
                                      input_tensor=Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
output_model = pre_trained_model.output
output_model = Flatten()(output_model)

print('10. Add new layers')
output_model = Dense(512, activation = 'relu')(output_model)
#output_model = Dropout(0.2)(output_model)
output_model = Dense(256, activation = 'relu')(output_model)
#output_model = Dropout(0.2)(output_model)
output_model = Dense(128, activation = 'relu')(output_model)
#output_model = Dropout(0.2)(output_model)
output_model = Dense(64, activation = 'relu')(output_model)
#output_model = Dropout(0.2)(output_model)
output_model = Dense(4, activation ='sigmoid')(output_model)

print('11. Add model parameters')
model = Model(inputs = pre_trained_model.input, outputs= output_model)
model.compile(loss='mse',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['mse'])
#model.summary()

# early_stop = EarlyStopping(monitor='val_loss', 
#                            patience=1, 
#                            verbose=1, 
#                            mode='auto')

print("## Finish creating the network ...\n")


# %%
# Fifth Step
print("## Start training the model ...")
print('10. Create TensorBoard')
tensorboard = TensorBoard('object detection')
print('11. Fit the model')
history = model.fit(x=x_train, 
                    y= y_train, 
                    batch_size=8, 
                    epochs=EPOCH, 
                    validation_data=(x_test,y_test), 
                    #callbacks=[early_stop, tensorboard])
                    callbacks=[tensorboard])
print('12. Save the model')
model_name = MODEL_DIR + '/' + MODEL_NAME
model.save(model_name)
print("## Finish training the model ...\n")

# %%
# Sixth Step
print("## Start plotting the model performance ...")
print('13. plotting the training and testing loss ...')
N = np.arange(0, EPOCH)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
#plt.plot(N, history.history["mse"], label="train_mse")
#plt.plot(N, history.history["val_mse"], label="val_mse")
plt.title("Training Loss and MSE (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/MSE")
plt.legend()
print("## End plotting the model performance ...")

# %%
# Seventh Step 
print('## Predict the street names ...')
test_path = os.path.join(os.getcwd(), TEST_DIR)
print(test_path)
test_images = os.listdir(test_path)
for index in range(len(test_images)):
    PerformOcr(TEST_DIR, PRED_DIR, test_images[index], model)
