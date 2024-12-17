
import tensorflow as tf
import numpy as np


def create_training_data():
    """Creates the data that will be used for training the model.

    Returns:
        (numpy.ndarray, numpy.ndarray): Arrays that contain info about the number of bedrooms and price in hundreds of thousands for 6 houses.
    """

    ### INIZIO CODICE ###

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float when defining the numpy arrays
    n_bedrooms =np.array([1.0,2.0,3.0,4.0, 5.0,6.0],dtype=float)

    price_in_hundreds_of_thousands =np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype=float)

    ### FINE CODICE ###

    return n_bedrooms, price_in_hundreds_of_thousands

features, targets = create_training_data()

print(f"Features have shape: {features.shape}")
print(f"Targets have shape: {targets.shape}")


def define_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict house prices.
    """

    ### INIZIO CODICE ###
    model = tf.keras.Sequential([

        # Definisco la "forma" dell'input (shape)
        tf.keras.Input(shape=(1,)),

        # Aggiungo uno solo strato
        ###  tf.keras.layers.Dense(units=1, activation='sigmoid')
        tf.keras.layers.Dense(units=1, activation='relu')
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    ### FINE CODICE ###

    return model

untrained_model = define_and_compile_model()

untrained_model.summary()


def train_model():
    """Returns the trained model.

    Returns:
        tf.keras.Model: The trained model that will predict house prices.
    """

    ### INIZIO CODICE ###

    # Addestramento modello (training)
    untrained_model.fit(features, targets, epochs=500)

    ### FINE CODICE ###

    return untrained_model

# Get your trained model
trained_model = train_model()
new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()
print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")