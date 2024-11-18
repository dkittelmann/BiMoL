################################ BACKPROPAGATION NETWORK ################################ 

# This script builds and trains a simple feedforward ANN employing backpropagation 
# on a statistical learning task mimicking the paradigm used in the study by 
# McDermott et al. 2024. 

# Author: Denise Kittelmann 


################################ LOAD PACKAGES ################################ 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json


################################ HELPER FUNCTIONS ################################ 

def tf_f_inv(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn == "LINEAR":
        m = x
    elif act_fn == "TANH":
        num = tf.ones_like(x) + x
        div = tf.ones_like(x) - x + 1e-7
        m = 0.5 * tf.math.log(num / div)
    elif act_fn == "LOGSIG":
        div = tf.ones_like(x) - x + 1e-7
        m = tf.math.log((x / div) + 1e-7)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def img_preproc(x, y, dtype=tf.float32): 
  """Cast input image to a certain tf dtype and normalize them between 0 and 1."""
  x = tf.cast(x, dtype) / 255.
  return x, y


def flatten(x, y):
  """flattens a video image series (or batch of images) to (n_batch, n_steps, 1) d."""
  shape = tf.shape(x)
  if len(shape) == 5: # hack, determining if it's a video or not
    x = tf.reshape(x, [shape[0], shape[1], -1])
  elif len(shape) == 4:
    x = tf.reshape(x, [shape[0], -1])
  return x, y

def augment_images(batch_images, batch_labels):
    """
    Applies data augmentation on a batch of images without TensorFlow Addons.
    
    Parameters:
    batch_images: Tensor of shape (n_batch, height, width, color channels)
    
    Returns:
    Augmented batch of images.
    """
    # Random horizontal flip
    augmented_images = tf.image.random_flip_left_right(batch_images)
    
    # Random brightness adjustment
    augmented_images = tf.image.random_brightness(augmented_images, max_delta=0.1)
    
    # Random contrast adjustment
    augmented_images = tf.image.random_contrast(augmented_images, lower=0.9, upper=1.1)
    
    # Random saturation adjustment
    augmented_images = tf.image.random_saturation(augmented_images, lower=0.9, upper=1.1)
    
    # Random hue adjustment
    augmented_images = tf.image.random_hue(augmented_images, max_delta=0.05)
    
    # Clipping to ensure pixel values are valid after transformations
    augmented_images = tf.clip_by_value(augmented_images, 0.0, 1.0)
    
    return augmented_images, batch_labels

def img_sequence(img_t1, img_t2, label_t1, label_t2, label_dict): 
    """This function stacks two images to construct an image pair and assigns a single label based on the label dictionary."""
    
    img_t1 = tf.cast(img_t1, dtype=tf.float32)
    img_t2 = tf.cast(img_t2, dtype=tf.float32)
    
    x = tf.concat([img_t1, img_t2], axis=0) 

    key_t1 = int(label_t1.numpy())
    key_t2 = int(label_t2.numpy())
    
    if (key_t1, key_t2) in label_dict:
        label = label_dict[(key_t1, key_t2)]
        #print(f"Label value found: {label}")
    else:
        print(f"Label pair {(key_t1, key_t2)} not found.")

    y = tf.cast(tf.random.uniform([]) < label, tf.float32)
    y = tf.expand_dims(y, axis=0)  
    
    return x,y

def generate_dataset(img_dirt1, img_dirt2, class_namest1, class_namest2, label_dict, image_size = None, seed = None, shuffle=False):       
    """ This function generates a dataset according to the labels defined in label dict. 
    
        PARAMETERS: imgdirt1, img_dirt2, class_namest1, class_namest2, label_dict, image_size = None, seed = None, shuffle=False
        
        RETURNS: tf.dataset with stacked image pairs along the height axis (i.e., axis 1)
        
    """
    
    data_t1 = tf.keras.preprocessing.image_dataset_from_directory(
        img_dirt1, 
        label_mode = 'int',
        class_names= class_namest1,
        batch_size = None,
        color_mode = 'rgb',
        image_size = image_size, 
        #shuffle = True, 
        seed = seed
        )

    data_t2 = tf.keras.preprocessing.image_dataset_from_directory(
        img_dirt2, 
        label_mode = 'int',
        class_names= class_namest2,
        batch_size = None,
        color_mode = 'rgb', 
        image_size = image_size, 
        #shuffle = True, 
        seed = seed
    )
    
    if shuffle:
        data_t1.shuffle(99999)
        data_t2.shuffle(99999)
    
    # Iterate through shuffled leading and trailing datasets
    leading = iter(data_t1)
    trailing = iter(data_t2) 
              
    while True:
        try:
            # Retrieve single samples
            img_t1, label_t1 = next(leading)
            img_t2, label_t2 = next(trailing)

            # Generate x, y pairs for single samples
            x, y = img_sequence(img_t1, img_t2, label_t1, label_t2, label_dict)
            yield x, y
            
        except StopIteration:
            # Break the loop if no more samples
            break



################################ DEFINE DATA PATHS ################################ 

img_dir_lead = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Leading/'
img_dir_trail = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Trailing/'
img_dir_test_lead = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Test/Test_Leading/'
img_dir_test_trail = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Test/Test_Trailing/'
class_names_L = ['barn', 'beach', 'cave', 'library', 'restaurant']
class_names_T = ['castle', 'Church', 'conference_room', 'forest'] 
batch_size = None # adjust if needed, e.g., 32
image_size = (28,28)
validation_split = 0.1
seed = 123


################################ CREATE LABEL DICT TO PREPARE THE DATASETS ################################ 

"""

Assign the correct labels for each leading-trailing img pair

L1 = barn = label 0 
L2 = beach = label 1
L3 = library = label 3
L4 = restaurant = label 4 
L5 = cave = label 2

T6 = Church = label 1   
T7 = conference room = label 2
T8 = castle = label 0   
T9 = forest = label 3

MAPPING:

L1 -> T6 = 0.75 -> (0,1) 
L1 -> T7 = 0.25 -> (0,2)
L1 -> T8 = 0 -> (0,0)
L1 -> T9 = 0 -> (0,3)

L2 -> T6 = 0.75 -> (1,1) 
L2 -> T7 = 0.25 -> (1,2)
L2 -> T8 = 0 -> (1,0)
L2 -> T9 = 0 -> (1,3)

L3 -> T6 = 0 -> (3,1) 
L3 -> T7 = 0 -> (3,2)
L3 -> T8 = 0.5 -> (3,0)
L3 -> T9 = 0.5 -> (3,3)

L4 -> T6 = 0.25 -> (4,1) 
L4 -> T7 = 0.75 -> (4,2)
L4 -> T8 = 0 -> (4,0)
L4 -> T9 = 0 -> (4,3)

L5 -> T6 = 0.25 -> (2,1) 
L5 -> T7 = 0.75 -> (2,2)
L5 -> T8 = 0 -> (2,0)
L5 -> T9 = 0 -> (2,3)

"""


label_dict = {
    (0, 1): 0.0,
    (0, 2): 0.75,
    (0, 0): 0.25,
    (0, 3): 0.25,
    
    (1, 1): 0.0,
    (1, 2): 0.75,
    (1, 0): 0.25,
    (1, 3): 0.25,
    
    (3, 1): 0.75,
    (3, 2): 0.75,
    (3, 0): 0.50,
    (3, 3): 0.50,
    
    (4, 1): 0.75,
    (4, 2): 0.0,
    (4, 0): 0.25,
    (4, 3): 0.25,
    
    (2, 1): 0.75, 
    (2, 2): 0.0,
    (2, 0): 0.25,
    (2, 3): 0.25
}

################################ CREATE VALIDATION DATASET ################################ 


print(f"################################ GENERATING THE VALIDATION SET ################################")

seed = 123 # to get the same validation set everytime we re-run it

val_dataset = tf.data.Dataset.from_generator(
    lambda: generate_dataset(img_dir_test_lead, img_dir_test_trail, class_names_L, class_names_T, label_dict, image_size = (28,28), seed = seed),
    output_signature=(
        tf.TensorSpec(shape=(56, 28, 3), dtype=tf.float32),  # shape of x 
        tf.TensorSpec(shape=(1), dtype=tf.float32)  # shape of y 
    )
) 

#print(val_dataset)


################################ MODEL TRAINING ################################ 

print(f"################################ DEFINING THE NETWORK NOW ################################")

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation="sigmoid"), # first hidden layer
    tf.keras.layers.Dense(units=4, activation="sigmoid"),  # second hidden layer 2
    tf.keras.layers.Dense(units=1, activation="sigmoid")  # output layer
])

model.build([None,4704]) # input öayer 
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-2),
              metrics=["accuracy"],
              loss="BinaryCrossentropy",  
              )

# save model before training
model.save('/Users/denisekittelmann/Documents/Python/BiMoL/results/bp_ann/final_model_bp_lr1e-7_untrained.keras') 

# evaluate accuracy 
loss, accuracy = model.evaluate(val_dataset.batch(512).map(img_preproc).map(flatten)) 
print(f"Accuracy before training: {accuracy}")


n_epochs = 1000

train_acc = []
val_acc = []
train_loss = []
val_loss = []
histories = []

print(f"################################ STARTING TO TRAIN THE NETWORK ################################")

# train the model and generate a new dataset each epoch 
for i in range(n_epochs): 
    
    resultspath = f'/Users/denisekittelmann/Documents/Python/BiMoL/results/bp_ann/model_checkpoint_{i:02d}_{{accuracy:.2f}}.keras'
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=resultspath,  
        monitor='accuracy',       
        save_best_only=False,      
        save_weights_only=False,   
        mode='max',                
        save_freq='epoch',         
        verbose=1                  
    )
    
    print(f"Epoch {i + 1}/{n_epochs}") 
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generate_dataset(img_dir_lead, img_dir_trail, class_names_L, class_names_T, label_dict, image_size = (28,28), seed = i, shuffle =True), 
        output_signature=(
            tf.TensorSpec(shape=(56, 28, 3), dtype=tf.float32),  # shape of x 
            tf.TensorSpec(shape=(1), dtype=tf.float32)  # shape of y 
        )
    )  
    

    
    history = model.fit(train_dataset.shuffle(99999).batch(512).map(img_preproc).map(augment_images).map(flatten), 
            validation_data=val_dataset.batch(512).map(img_preproc).map(flatten),
            callbacks=[checkpoint_callback]) 

    # have history throughout training
    train_acc.extend(history.history['accuracy'])
    val_acc.extend(history.history['val_accuracy'])
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    
    epoch_history = {
        "epoch": i + 1,
        "history": history.history
    }
    
    histories.append(epoch_history)

# save history to a single JSON file
    with open('/Users/denisekittelmann/Documents/Python/BiMoL/results/bp_ann/epochs_history.json', 'w') as f:
        json.dump(histories, f)


loss, accuracy = model.evaluate(val_dataset.batch(512).map(img_preproc).map(flatten)) 
print(f"Accuracy after training: {accuracy}")

print(f"################################ SAVING THE FINAL MODEL AFTER TRAINING ################################")
model.save('/Users/denisekittelmann/Documents/Python/BiMoL/results/bp_ann/final_model_bp_lr1e-7_trained.keras') 



################################ PLOT ACCURACY COLLECTED DURING TRAINING ################################ 

plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()