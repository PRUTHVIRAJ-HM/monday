import numpy as np
with open("corpus.txt","r",encoding="utf-8")as f:
    documents=f.readlines()
    

corpus=" ".join(doc.strip().lower() for doc in documents)
words=corpus.split()
vocab=list(set(words))
word2idx={w:i for i,w in enumerate(vocab)}
idx2word={i:w for w,i in word2idx.items()}

window_size=2
pairs=[]

for i,word in enumerate(words):
    center=word2idx[word]
    for j in range(max(0,i-window_size),min(len(words),i+window_size+1)):
        if i!=j:
            context=word2idx[words[j]]
            pairs.append((center,context))

def one_hot(idx,size):
    vec=np.zeros(size)
    vec[idx]=1
    return vec

x_train=np.array([one_hot(c,len(vocab)) for c,_ in pairs])
y_train=np.array([one_hot(ctx,len(vocab)) for _,ctx in pairs])

embedding_dim=5

w1=np.random.randn(len(vocab),embedding_dim)*0.01
w2=np.random.randn(embedding_dim,len(vocab))*0.01

def softmax(x):
    e=np.exp(x-np.max(x))
    return e/e.sum(axis=0)

lr=0.05
epochs=300

for epoch in range(epochs):
    loss=0
    for x,y in zip(x_train,y_train):
        h=np.dot(x,w1)
        u=np.dot(h,w2)
        y_pred=softmax(u)

        loss-=np.sum(y*np.log(y_pred+1e-9))

        e=y_pred-y
        dw2=np.outer(h,e)
        dw1=np.outer(x,np.dot(w2,e))

        w1-=lr*dw1
        w2-=lr*dw2
    
    if(epoch+1)%100==0:
        print(f"Epoch {epoch+1}, Loss:{loss:.4f}")

embeddings=w1

print("\nWord Embeddings:")
for word,idx in word2idx.items():
    print(f"{word}:{[int(w*10000)/10000 for w in embeddings[idx]]}")

def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-9)

def most_similar(word,topn=3):
    if word not in word2idx:
        return[]
    idx=word2idx[word]
    vec=embeddings[idx]
    sims=[]
    for w,i in word2idx.items():
        if w==word:
            continue
        sims.append((w,cosine_similarity(vec,embeddings[i])))
        sims.sort(key=lambda x:x[1],reverse=True)
        return sims[:topn]
    
print("\nMost similar word:")
print("Learning->",most_similar("learning"))
print("Language->",most_similar("language"))
print("neural->",most_similar("neural"))
