from collections import defaultdict

reviews = [
    ("fun, couple, love, love", "comedy"),
    ("fast, furious, shoot", "action"),
    ("couple, thy, fast, fum", "comedy"),
    ("furious, shoot, shoot, fun", "action"),
    ("fly, fast, shoot, love", "action")
]

class_word_counts = defaultdict(lambda: defaultdict(int))
class_counts = defaultdict(int)
vocab = set()

for review, label in reviews:
    words = review.split(",")
    class_counts[label] += 1
    for word in words:
        word = word.strip()
        class_word_counts[label][word] += 1
        vocab.add(word)

V = len(vocab)

prior_comedy = class_counts["comedy"] / sum(class_counts.values())
prior_action = class_counts["action"] / sum(class_counts.values())

def calc_likelihoods(class_word_counts, class_label):
    total_words = sum(class_word_counts[class_label].values())
    likelihoods = {}
    for word in vocab:
        likelihoods[word] = (class_word_counts[class_label].get(word, 0) + 1) / (total_words + V)
    return likelihoods

likelihood_comedy = calc_likelihoods(class_word_counts, "comedy")
likelihood_action = calc_likelihoods(class_word_counts, "action")

new_document = "fast, couple, shoot, fly"
new_words = new_document.split(",")

posterior_comedy = prior_comedy
posterior_action = prior_action

for word in new_words:
    word = word.strip()
    posterior_comedy *= likelihood_comedy[word]
    posterior_action *= likelihood_action[word]

print(f"Posterior for comedy: {posterior_comedy}")
print(f"Posterior for action: {posterior_action}")

if posterior_comedy > posterior_action:
    print("The most likely class for the document is: comedy")
else:
    print("The most likely class for the document is: action")