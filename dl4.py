
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input,Dense
from keras.optimizers import Adam

(x_train,_),(x_test,_)=mnist.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

x_train=x_train.reshape((len(x_train),28*28))
x_test=x_test.reshape((len(x_test),28*28))

input_dim=28*28
encoding_dim=32
input_img=Input(shape=(input_dim,))
encoded=Dense(128,activation='relu')(input_img)
encoded=Dense(64,activation='relu')(encoded)
code=Dense(encoding_dim,activation='relu')(encoded)

decoded=Dense(64,activation='relu')(code)
decoded=Dense(128,activation='relu')(decoded)
decoded=Dense(input_dim,activation='sigmoid')(decoded)

autoencoder=Model(input_img,decoded)
encoder=Model(input_img,code)

autoencoder.compile(optimizer=Adam(learning_rate=0.001),loss='mse')
history=autoencoder.fit(
    x_train,x_train,epochs=20,batch_size=256,shuffle=True,validation_data=(x_test,x_test)
)

enocoded_imgs=encoder.predict(x_test)
decoded_imgs=autoencoder.predict(x_test)

plt.figure()
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Autoencoder Training loss")
plt.show()

n=5
plt.figure(figsize=(10,4))

for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28),cmap='gray')
    plt.title("Original")
    plt.axis('off')

    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28),cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()
print(f"input dim={input_dim},code dim={encoding_dim}")
print(f"Compression ration={input_dim/encoding_dim:.2f}x")