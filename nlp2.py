from collections import Counter
c=[
    "The sun rises in the east",
    "Birds fly in the sky",
    "The sky is blue",
    "The moon is bright",
    "Birds are flying in the sky",
    "The moon rises in the east"
]

def pc(c):
    ts=[]
    for s in c:
        tokens=["<start>"]+s.lower().split()+["<end>"]
        ts.append(tokens)
    return ts

def cng(ts,n):
    ngrams=[]
    contexts=[]
    for s in ts:
        for i in range(len(s)-n+1):
            ngr=tuple(s[i:i+n])
            context=tuple(s[i:i+n-1])
            ngrams.append(ngr)
            contexts.append(context)
    return ngrams, contexts

def prob_smooth(ngrams,contexts,v):
    n_counts=Counter(ngrams)
    c_counts=Counter(contexts)
    prob={}
    for ngram, count in n_counts.items():
        context=ngram[:-1]
        context_count=c_counts[context]
        prob_ng=(count+1)/(context_count+v)
        prob[ngram]=prob_ng
    return prob,n_counts,c_counts

def sent_prob(sentence,ng_prob,n,ng_counts,c_counts,v):
    proc_sen=["<start>"]+sentence.lower().split()+["<end>"]
    if len(proc_sen)<n:
        print("Sentence is too short")
        return 0.0
    tot_prob=1.0
    for i in range(len(proc_sen)-n+1):
        ngram=tuple(proc_sen[i:i+n])
        if n==1:
            tot_words=sum(ng_counts.values())
            prob_ngram=ng_prob.get(ngram,0)/tot_words
        else:
            prob_ngram=ng_prob.get(ngram,0.0)
        
        if prob_ngram==0.0:
            print(f"warning: N-gram '{ngram}' not found. Applying smoothing.")
            prob_ngram=(1/(sum(c_counts.values())+v))
        tot_prob*=prob_ngram
    return tot_prob

tc=pc(c)
v=len(set([word for sentence in tc for word in sentence]))
print("---Unigram Model (n=1)---")
u_ngram,u_cont=cng(tc,1)
u_prob,u_ngcounts,u_contextcount=prob_smooth(u_ngram,u_cont,v)
print("unigram probabilities")
print(u_prob)

s_to_test="The sun rises"
prob_sen_u=sent_prob(s_to_test,u_prob,1,u_ngcounts,u_contextcount,v)
print(f"Probability of '{s_to_test}' (unigram):{prob_sen_u:.10f}\n")

print("---Bigram Model (n=2)---")
b_ngram,b_cont=cng(tc,2)
b_prob,b_ngcounts,b_contextcount=prob_smooth(b_ngram,b_cont,v)
print("bigram probabilities")
print(b_prob)

prob_sen_b=sent_prob(s_to_test,b_prob,1,b_ngcounts,b_contextcount,v)
print(f"Probability of '{s_to_test}' (bigram):{prob_sen_b:.10f}\n")

print("---trigram Model (n=2)---")
t_ngram,t_cont=cng(tc,3)
t_prob,t_ngcounts,t_contextcount=prob_smooth(t_ngram,t_cont,v)
print("trigram probabilities")
print(t_prob)

prob_sen_t=sent_prob(s_to_test,b_prob,1,b_ngcounts,b_contextcount,v)
print(f"Probability of '{s_to_test}' (trigram):{prob_sen_t:.10f}\n")