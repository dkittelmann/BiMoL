################################ EXTRACT MEAN NETWORK ACTIVATIONS ################################ 

# This scripts extracts condition-specific network activations.

# Author: Denise Kittelmann 

################################ LOAD REQUIRED PACKAGES ################################ 

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
from scipy.spatial.distance import cosine
import zipfile
import itertools


################################ DEFINE PATHS ################################ 


# Model paths
dir_bpann = "/Users/denisekittelmann/Documents/Python/BiMoL/results/bp_ann/model_checkpoint_649_0.67.keras"
dir_pcn = "/Users/denisekittelmann/Documents/Python/BiMoL/results/pcn/model_checkpoint_pcnoriginal_726_0.67.keras"
unzip_path = "/Users/denisekittelmann/Documents/Python/BiMoL/results/pcn/pcn_tt/"


# Image paths
img_dir_lead = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Leading/'
img_dir_trail = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Trailing/'
img_dir_test_lead = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Test/Test_Leading/'
img_dir_test_trail = '/Users/denisekittelmann/Documents/Python/BiMoL/data/Test/Test_Trailing/'
class_names_L = ['barn', 'beach', 'cave', 'library', 'restaurant']
class_names_T = ['castle', 'Church', 'conference_room', 'forest'] # changed the order 
batch_size = None # adjust if needed, e.g., 32
image_size = (28,28)
validation_split = 0.1
seed = 123

# Results path
savepath = "/Users/denisekittelmann/Documents/Python/BiMoL/results/"


################################ DEFINE NECESSARY DICTS ################################ 

label_dict = {
    (0, 1): 0.0, (1, 1): 0.0,
    (3, 2): 0.0, (4, 2): 0.0,
    (0, 2): 0.75, (1, 2): 0.75,
    (3, 1): 0.75, (4, 1): 0.75,
    (2, 0): 0.5,
    (2, 3): 0.5
}

lead_cat = {
    "barn": 0,
    "beach": 1,
    "cave": 2,
    "library": 3,
    "restaurant": 4
}

trail_cat = {
    "church": 1,
    "castle": 0,
    "conference_room": 2,
    "forest": 3
}

rdm_dict = {
    0: ((0, 1), (1, 1)),  # 0.75%: barn -> church, beach -> church
    1: ((3, 2), (4, 2)),  # 0.75%: library -> conference room, restaurant -> conference room
    2: ((0, 2), (1, 2)),  # 0.25%: barn -> conference room, beach -> conference room
    3: ((3, 1), (4, 1)),  # 0.25%: library -> church, restaurant -> church
    4: ((2, 0),),         # 0.5%: cave -> castle
    5: ((2, 3),)          # 0.5%: cave -> forest
}

################################ HELPER FUNCTIONS TO GENERATE DATASET ################################ 


def generate_possible_img_pairings(rdm_dict, category_id, img_dirt1, img_dirt2, class_namest1, class_namest2, label_dict, image_size=(28, 28), seed=None):
    '''This function generates a dataset with all possible image combinations based on rdm_dict.'''

    data_t1 = tf.keras.preprocessing.image_dataset_from_directory(
        img_dirt1,
        label_mode='int',
        class_names=class_namest1,
        batch_size=None,
        color_mode='rgb',
        image_size=image_size,
        seed=seed
    )

    data_t2 = tf.keras.preprocessing.image_dataset_from_directory(
        img_dirt2,
        label_mode='int',
        class_names=class_namest2,
        batch_size=None,
        color_mode='rgb',
        image_size=image_size,
        seed=seed
    )


    l_data_t1 = list(data_t1)
    l_data_t2 = list(data_t2)


    x_pairs = []
    y_labels = []

    # Generate pairs for the specified category_id
    if category_id in rdm_dict:
        pairs = rdm_dict[category_id]
        for lead, trail in pairs:
            # Filter images by label in data_t1 and data_t2 for the current lead and trail categories
            lead_images = [img for img, lbl in l_data_t1 if lbl.numpy() == lead]
            trail_images = [img for img, lbl in l_data_t2 if lbl.numpy() == trail]

            # Generate all possible combinations of image pairs according to current leading and trailing image category
            for img_t1, img_t2 in itertools.product(lead_images, trail_images):
                # Concatenate images along the height dimension -> (56, 28, 3)
                x = tf.concat([img_t1, img_t2], axis=0)  
                label = label_dict[(lead, trail)] 
                y = tf.cast(tf.random.uniform([]) < label, tf.float32) 
                y = tf.expand_dims(y, axis=0) 

                x_pairs.append(x)  
                y_labels.append(y)  

    # Concatenate all pairs and the corresponding labels  
    x_stacked = tf.concat([tf.expand_dims(x, axis=0) for x in x_pairs], axis=0)
    y_stacked = tf.concat([tf.expand_dims(y, axis=0) for y in y_labels], axis=0)
    
    return x_stacked, y_stacked  


################################ ANN HELPER FUNCTIONS ################################ 
  
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
  #x = tf_scale_imgs(x, cf.img_scale)
  #y = tf_scale_labels(y, cf.label_scale)
  #x = tf_f_inv(x, "TANH")
  #y = tf.one_hot(y, depth=10)
  return x, y


def flatten(x, y):  
  #flattens a video image series (or batch of images) to (n_batch, n_steps, 1) d.
  shape = tf.shape(x)
  if len(shape) == 5: # hack, determining if it's a video or not (batch_size, n_steps, height, width, channels)
    x = tf.reshape(x, [shape[0], shape[1], -1])
  elif len(shape) == 4: # regular image (batch_size, height, width, channels)
    x = tf.reshape(x, [shape[0], -1])
  elif len(shape) == 3:  # Single image (height, width, channels)
    x = tf.reshape(x, [-1])
  return x, y


################################ RELOAD PCN ################################ 

# To reload the PCN, we need to call the model once again and nee the specific customed Dense layer model itself. 

class CustomDense(tf.keras.layers.Dense):
    def call(self, inputs):
        """This works like a dense, except for the activation being called earlier."""
        # Apply the activation to the input first
        activated_input = self.activation(inputs)
        # Perform the matrix multiplication and add the bias
        output = tf.matmul(activated_input, self.kernel)
        if self.use_bias:
            output = output + self.bias
        return output


class PredictiveCodingNetwork(tf.keras.Sequential):
    def __init__(self, layers, vars, beta, **kwargs):
        """Initialize a PredictiveCodingNetwork"""
        super().__init__(layers, **kwargs)
        self.vars = tf.convert_to_tensor(vars, dtype=tf.float32)
        self.beta = beta

    def call_with_states(self, x):
        x_list = [x]
        for layer in self.layers:
            x = layer(x)
            x_list.append(x)
        return x_list

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # do the stuff we do in train_epochs
        outputs, errors = self.infer(x, y)
        self.update_params(outputs, errors)

        # Update metrics (includes the metric that tracks the loss)
        pred = self.call(x)
        for metric in self.metrics:
            metric.update_state(y, pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
   
    def infer(self, x_batch, y_batch=None, n_iter=50, return_sequence=False):
        """Note: while model call, call with states and model evaluate take
        2D input, train_step and infer take stacked 3D inputs."""
        if return_sequence:
            errors_time = []
            states_time = []
        errors = [None for _ in range(len(self.layers))]
        f_x_arr = [None for _ in range(len(self.layers))]
        f_x_deriv_arr = [None for _ in range(len(self.layers))]
        shape = x_batch.shape
        batch_size = shape[0]

        for itr in range(n_iter):
            # if its the first itr, set x to the current forward call
            if itr == 0:
                x = self.call_with_states(x_batch)

                if y_batch is not None:
                  x[-1] = y_batch
            else:
                # update g and x only for consecutive iterations
                for l in range(1, len(self.layers)):
                    g = tf.multiply(tf.matmul(errors[l], self.layers[l].kernel, transpose_b=True), f_x_deriv_arr[l])
                    x[l] = x[l] + self.beta * (-errors[l-1] + g)

            # update f_x etc for every iteration
            for l in range(len(self.layers)):
                f_x = self.layers[l].activation(x[l])
                f_x_deriv_fn = self.get_activation_derivative(self.layers[l].activation)
                f_x_deriv = f_x_deriv_fn(x[l])
                f_x_arr[l] = f_x
                f_x_deriv_arr[l] = f_x_deriv
                errors[l] = (x[l + 1] - tf.matmul(f_x, self.layers[l].kernel) - self.layers[l].bias) / self.vars[l]
            
            if return_sequence:
                errors_time.append(errors)
                states_time.append(x)

        # return what we want to return
        if return_sequence:
            states_time = [tf.stack(tensors, axis=1) for tensors in zip(*states_time)]
            errors_time = [tf.stack(tensors, axis=1) for tensors in zip(*errors_time)]
            return states_time, errors_time
        else:
            return x, errors
    
    # We need to check if we actually need call here.
    # Now, call will give us the result of the network after the first inference step
    # If we want to have the results after the last inference step, we would need to change this
    #def call(self, inputs, training=False):
    #    """Call, but time distributed."""
    #    x, errors = self.infer(inputs, return_sequence=False)
    #    return x[-1]

    def update_params(self, x, errors):
        """Update the model parameters."""
        batch_size = tf.cast(tf.shape(x[0])[0], tf.float32)
        gradients = []
        for l, layer in enumerate(self.layers):
            grad_w = self.vars[-1] * (1 / batch_size) * tf.matmul(tf.transpose(self.layers[l].activation(x[l])), errors[l])
            grad_b = self.vars[-1] * (1 / batch_size) * tf.reduce_sum(errors[l], axis=0)
            gradients.append((-grad_w, layer.kernel))
            gradients.append((-grad_b, layer.bias))
        self.optimizer.apply_gradients(gradients)

    def get_activation_derivative(self, activation):
        """Return a function for the derivative of the given activation function."""
        activation_fn = tf.keras.activations.get(activation)
        if activation_fn == tf.keras.activations.linear:
            return lambda x: tf.ones_like(x)
        elif activation_fn == tf.keras.activations.tanh:
            return lambda x: 1 - tf.square(tf.nn.tanh(x))
        elif activation_fn == tf.keras.activations.sigmoid:
            return lambda x: tf.nn.sigmoid(x) * (1 - tf.nn.sigmoid(x))
        else:
            raise ValueError(f"{activation} not supported")
 
# To call backprop activations        
def call_with_states(model, x):
     outputs = []
     for layer in model.layers:
          x = layer(x)
          outputs.append(x)
     return outputs
 
 


################################ RELOAD ANNs ################################ 

input_layer = tf.keras.layers.Input(shape=(4704,))

model = PredictiveCodingNetwork([CustomDense(units=6, activation="sigmoid"),
                                 CustomDense(units=4, activation="sigmoid"), 
                                 CustomDense(units=1, activation="sigmoid")], 
                                vars=[1, 1, 1], # variances. This is super useless and in the code only the last variance is used
                                beta=0.1)

keras_file_path = "/Users/denisekittelmann/Documents/Python/BiMoL/results/pcn/model_checkpoint_pcnoriginal_726_0.67.keras"
unzip_path = "/Users/denisekittelmann/Documents/Python/BiMoL/results/pcn/pcn_tt/"

with zipfile.ZipFile(keras_file_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)


pcn = model 

pcn.build([None, 4704])
pcn.load_weights("/Users/denisekittelmann/Documents/Python/BiMoL/results/pcn/pcn_tt/model.weights.h5")

pcn.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-7, weight_decay=1e-2),
    loss="categorical_crossentropy",  # Placeholder loss
    metrics=["accuracy"]
)


# Backprop ANN
bp_ann = tf.keras.models.load_model(dir_bpann)

# Some random checks 
#pcn.infer(tf.random.normal([64, 4704]), return_sequence=True)
#[[i.shape for i in j] for j in pcn.infer(tf.random.normal([64, 4704]), return_sequence=True)]
# bp_ann(tf.random.normal([64, 4704]))


################################ EXTRACT ACTIVATIONS ################################ 


pcn_error_mean_activations = {condition: [] for condition in rdm_dict.keys()}
pcn_layer_mean_activations = {condition: [] for condition in rdm_dict.keys()}
bp_layer_mean_activations = {condition: [] for condition in rdm_dict.keys()}


for category_id in rdm_dict.keys():
    print(f"Processing category: {category_id}")

    # Generate all possible category-specific image pairs
    val_dataset = generate_possible_img_pairings(
        rdm_dict, category_id, img_dir_test_lead, img_dir_test_trail, class_names_L, class_names_T, label_dict, image_size=(28, 28), seed=seed
    )
    
    # Flatten images and only return images
    val_dataset = flatten(*img_preproc(val_dataset[0], val_dataset[1]))[0]
    
    # Generate PCN activations & errors
    pcn_acts, pcn_err = pcn.infer(val_dataset, return_sequence=True)
    
    pcn_acts = [tf.reshape(tensor, (tensor.shape[0], -1)) for tensor in pcn_acts] # flatten for later concat of all layers in the next step
    pcn_err = [tf.reshape(tensor, (tensor.shape[0], -1)) for tensor in pcn_err]


    # Generate BP ANN activations
    bp_acts = call_with_states(bp_ann, val_dataset)

    # Compute mean PCN error across all layers
    mean_error_all_layers = np.mean(np.concatenate([layer for layer in pcn_err], axis=1), axis=0)
    pcn_error_mean_activations[category_id].append(mean_error_all_layers)

    # Compute mean PCN activations across all layers (without the first layer aka input layer)
    mean_activations_pcn = np.mean(np.concatenate([layer for layer in pcn_acts[1:]], axis=1), axis=0)
    pcn_layer_mean_activations[category_id].append(mean_activations_pcn)

    # Compute mean BP ANN activations across all layers (without the first layer aka input layer)
    mean_activations_bp = np.mean(np.concatenate([layer for layer in bp_acts[1:]], axis=1), axis=0)
    bp_layer_mean_activations[category_id].append(mean_activations_bp)
    

# Calculate grand mean conditions 
final_pcn_error_mean_activations = {condition: np.mean(activations, axis=0) for condition, activations in pcn_error_mean_activations.items()}
final_pcn_layer_mean_activations = {condition: np.mean(activations, axis=0) for condition, activations in pcn_layer_mean_activations.items()}
final_bp_layer_mean_activations = {condition: np.mean(activations, axis=0) for condition, activations in bp_layer_mean_activations.items()}


################################ CALCULATE GRAND GRAND MEAN ################################ 

final_pcn_error_mean_activations_final = {
    category: np.mean(values)  
    for category, values in final_pcn_error_mean_activations.items()
}


final_pcn_acts_mean_activations_final = {
    category: np.mean(values)  
    for category, values in final_pcn_layer_mean_activations.items()
}

final_bp_acts_mean_activations_final = {
    category: np.mean(values)  
    for category, values in final_bp_layer_mean_activations .items()
}

# Save as np arrays
np.save(f"{savepath}/final_pcn_error_mean_activations_final.npy", final_pcn_error_mean_activations_final)
np.save(f"{savepath}/final_pcn_acts_mean_activations_final.npy", final_pcn_acts_mean_activations_final)
np.save(f"{savepath}/final_bp_acts_mean_activations_final.npy", final_bp_acts_mean_activations_final)