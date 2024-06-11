import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 訓練和驗證數據的路徑
train_dir = 'animals_bird'  # 更新為你的訓練數據路徑
val_dir = 'animals_bird'  # 更新為你的驗證數據路徑

required_classes = ['birds', 'cats', 'dogs', 'panda']

# 檢查驗證數據目錄
# 檢查驗證數據目錄
if not os.path.exists(val_dir):
    raise ValueError(f"驗證數據目錄 {val_dir} 不存在")

# 檢查每個類別子目錄是否存在
for class_name in required_classes:
    class_dir = os.path.join(val_dir, class_name)
    if not os.path.exists(class_dir):
        raise ValueError(f"驗證數據目錄中缺少類別子目錄: {class_name}")
    if not os.listdir(class_dir):
        raise ValueError(f"類別子目錄 {class_name} 中沒有圖像文件")

batch_size = 32
img_size = (224, 224)  # ResNet-50 要求的輸入尺寸

# 數據增強和預處理
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

if train_generator.num_classes != 4:
    raise ValueError("訓練數據目錄中的類別數量不為 4")
if val_generator.num_classes != 4:
    raise ValueError("驗證數據目錄中的類別數量不為 4")

# 建立模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # 4 個類別

model = Model(inputs=base_model.input, outputs=predictions)

# 冷凍 base_model 的所有層
for layer in base_model.layers:
    layer.trainable = False

# 編譯模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
epochs = 10

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# 繪製訓練過程中的準確率和損失值
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend(loc='upper left')
plt.show()

# 解凍 base_model 的一些層來進行微調
for layer in base_model.layers[-20:]:
    layer.trainable = True

# 再次編譯模型
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 繼續訓練模型
fine_tune_epochs = 20
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator
)

# 繪製微調過程中的準確率和損失值
plt.plot(history_fine.history['accuracy'], label='accuracy')
plt.plot(history_fine.history['val_accuracy'], label='val_accuracy')
plt.plot(history_fine.history['loss'], label='loss')
plt.plot(history_fine.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend(loc='upper left')
plt.show()
# 保存模型
model.save('animal_efficient.hdf5')