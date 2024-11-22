from keras.src.models import Model
from keras.src.layers import Dense, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
import keras

# Load VGG16 without the top layer (fully connected layers)
base_model = keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(6, activation='softmax')(x)  # Assuming 2 classes: fresh and rotten

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing
trdata = ImageDataGenerator(rescale=1.0/255)
traindata = trdata.flow_from_directory(
    directory="/home/mw/.cache/kagglehub/datasets/sriramr/fruits-fresh-and-rotten-for-classification/versions/1/dataset/train/",
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

tsdata = ImageDataGenerator(rescale=1.0/255)
testdata = tsdata.flow_from_directory(
    directory="/home/mw/.cache/kagglehub/datasets/sriramr/fruits-fresh-and-rotten-for-classification/versions/1/dataset/test/",
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    traindata,
    validation_data=testdata,
    epochs=25,
    callbacks=[checkpoint, early_stop]
)

# Evaluate the model
loss, accuracy = model.evaluate(testdata)
print(f"Test Accuracy: {accuracy:.2f}")
