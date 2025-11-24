# Estimating Performance

We want to answer the following question: "Given my code and my chip, what is the fastest theoretical time this function should take?"

We can simply model it, by assuming we can overlap all the components involved in the operation (Memory, Tensor Cores, etc), the theoretical fastest time is going to be the time of the slowest component.
