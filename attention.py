import numpy as np

def scaled_dot_product_attention(Q, K, V):
    scores = np.dot(Q, K.T)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context_vector = np.dot(attention_weights, V)

    return attention_weights, context_vector

# Example run
Q = np.array([[1, 0, 1]])
K = np.array([[1, 0, 1],
              [0, 1, 0]])
V = np.array([[1, 2, 3],
              [4, 5, 6]])

weights, context = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", weights)
print("Context Vector:\n", context)
