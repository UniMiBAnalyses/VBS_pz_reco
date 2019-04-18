'''

Models File, there are many different combination of hidden layers, neurons, activation function, loss and metrics



Christian Uccheddu

'''
from keras.models import Sequential, load_model
from keras.activations import relu
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from keras import losses


def getModel(id, input_dim):

    print(">>> Creating model...")
    model = Sequential()



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        RETI USATE    <<<<<<<<<<<<<<<<<<<<<<<<<            

    if id=="mse_four_hidden_layer_400":

        model.add(Dense(units=400,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])


    elif id=="mse_three_hidden_layer_400":

        model.add(Dense(units=400,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])

    elif id=="mse_three_hidden_layer_300":

        model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=300,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=300,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=300,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])


    elif id=="mse_three_hidden_layer_200":

        model.add(Dense(units=200,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])


    elif id=="mse_three_hidden_layer_100":

        model.add(Dense(units=100,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])          
    
    elif id=="mse_three_hidden_layer_50":

        model.add(Dense(units=50,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error]) 

    elif id=="mse_three_hidden_layer_10":

        model.add(Dense(units=10,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=10,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=10,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=10,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])    


    elif id=="mse_ten_hidden_layer_15":

        model.add(Dense(units=15,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])


    elif id=="mse_three_hidden_layer_15":

        model.add(Dense(units=15,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])

    elif id=="mse_four_hidden_layer_15":

        model.add(Dense(units=15,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])

    elif id=="mse_two_hidden_layer_400":

        model.add(Dense(units=400,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dropout(0.2))
        


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])


    elif id=="mse_fifteen_hidden_layer_15":

        model.add(Dense(units=15,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=15,activation="relu"))
        model.add(Dropout(0.2))


        model.add(Dense(1))
        
        model.compile(loss=losses.mean_squared_error,
              optimizer='adam',
              metrics=[metrics.mean_absolute_error])

    return model