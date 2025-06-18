# LLM From Scratch — My Learning Journey

Welcome to my **LLM Practice** repository! 

In this repo, I’m exploring how **Large Language Models (LLMs)** work by building components from scratch and experimenting with core concepts like tokenization, embeddings, attention, transformers, and more.

Rather than just using pre-built tools, my goal is to **understand the internals step by step** — how each part functions and how everything fits together.

###  What You’ll Find Here

This `README.md` will serve as a living document — a kind of **learning journal** — where I’ll write down the key concepts I explore, what I understood, and small explanations in my own words.

Whenever I dive into a new concept, such as embedding layers, positional encodings, self-attention, or model training, I’ll update this file with:

- A beginner-friendly explanation
- Code snippets I’ve written or modified
- My personal observations and insights

---

>  The goal of this project is not just to build something that works, but to **learn deeply** by experimenting, breaking things, fixing them, and documenting the journey.

So far, here’s what I’ve learned:


##
Understanding Embedding Layers — What I Learned

While working on my LLM practice, I wanted to dive deep into how embedding layers actually work. At first, I wondered: **if embeddings only take integers as input, how do they capture the meaning of words?**

Here’s what I found out, and it’s pretty cool:

### What is an Embedding Layer?

Imagine you have a vocabulary of words, and each word is assigned a unique number (an index). The embedding layer’s job is to turn those numbers into meaningful vectors — lists of numbers that represent the words in a way a machine learning model can understand.

### But wait — if the input is just integers, how does it know the *meaning*?

Great question! At the start, the embedding layer is basically a big table filled with random vectors, one vector for each word index. When the model trains on real data (say, predicting the next word, or classifying sentiment), it adjusts these vectors little by little to reduce errors.

Because words that appear in similar contexts often help the model solve the task similarly, their vectors gradually become closer to each other in this multi-dimensional space.

So, **the meaning of words emerges as the model learns** — even though all it ever sees as input are just integer indices.

### In short:

- The embedding layer maps word indices to vectors.
- These vectors start random.
- Training updates the vectors so that similar words have similar vectors.
- This helps the model understand semantic relationships between words.

---

This understanding really helped me see why embeddings are so powerful and a key building block in NLP models!

