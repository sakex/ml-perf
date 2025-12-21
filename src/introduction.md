# Introduction

This is the book I which I had when I made my transition from generalist Software Engineer to ML engineer at Deepmind.

I started it as interview preparation, but it quickly evolved into a more comprehensive list of skills I acquired during my time at Google Deepmind. A lot of the skills required in ML performance are scattered around in different resources, or have to be learnt on the job by talking to more experienced engineers. This book tries to assemble the most important bits in a single place.

I use Gemini extensively to refine my words (I am not a native english speaker) and to generate diagrams (with Nano Banana 3.)

## Goals and Non-Goals

The book is an introduction to the most important concepts required to succeed in ML engineering. Its goal is to cover a large breadth of subjects but not to dive too deep into any single one. Most of the topics discussed are active research problems with new papers being published frequently. Building a T-shaped skillset - good knoweldge in a lot of subjects and expert knowledge in a handful of others - is often recommended for a successful career. Readers are encouraged to go and read the latest papers in the subjects that interested them the most.

## Prerequisites

A good understanding of computer programming in Python is required. Linear algebra, Machine Learning, and distributed programming skills will greatly help understanding the material but are not necessary.

## Structure

The book gradually introduces concepts that build on top of each other as the chapters go by. We first introduce the basic APIs that are commonly used to build ML models. After that, we have a small chapter discussing the backward pass and its performance implications. Then, we discuss concurrency on modern hardware, how to leverage the different levels of concurrency, and how to think about and estimate performance. This leads us up to discussing multi-device distributed computations, what are the primitive operations and the common strategies for distributing ML models. We finish by introducing commonly used techniques that leverage everything we have discussed to serve LLMs efficiently at scale.

## Playing Along

We try to add code examples whenever possible. Feel free to copy-paste to a Jupyter Notebook such as [Google Colab](https://colab.google.com) or into your code editor to run the code and play with it. *Some code examples in the [Distributed section](./distributed/distributed.html) are conceptual pseudocode only and therefore will not work by themselves.*

## Contributing

Contributions to the [GitHub Repository](https://github.com/sakex/ml-perf) would be very much appreciated!
