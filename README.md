# 700762701-HW-5
# Attentionproject
SREE RAM CHARAN TEJA PUDARI 700762701
This file contains both readme files for question1&question2
## **📌 Project Overview**

This project implements the **Scaled Dot-Product Attention** mechanism using **NumPy**, following the formula used in Transformer models.

The purpose of this script is to:

* Compute attention scores
* Apply scaling
* Normalize using softmax
* Generate attention weights
* Produce the final context vector

This is a core building block of **Multi-Head Attention** and Transformer models.

## **📁 Files Included**

* **attention.py**
  Contains the full implementation of the scaled dot-product attention function along with an example test run.

## **🧠 Formula Used**

The attention is calculated using:

[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Where:

* **Q** = Query
* **K** = Key
* **V** = Value
* **d_k** = Dimension of the key vector

---

## **🔧 Requirements**

Make sure Python and NumPy are installed:

```bash
pip install numpy
```

---

## **▶️ How to Run**

1. Open the project folder in VS Code
2. Open a terminal inside VS Code
3. Run the script:

```bash
python3 attention.py
```

---

## **💡 Sample Output**

Example output from the provided test:

```
Attention Weights:
 [[0.76036844 0.23963156]]

Context Vector:
 [[1.71889467 2.71889467 3.71889467]]
```

This shows:

* The model attends ~76% to the first key-value pair
* ~24% to the second
* The resulting context vector is the weighted sum of V

---

## **📜 Code Summary**

The script:

1. Computes the dot product **QKᵀ**
2. Scales by **√dₖ**
3. Applies **softmax**
4. Computes **Context = AttentionWeights × V**

The main function:

```python
def scaled_dot_product_attention(Q, K, V):
    scores = np.dot(Q, K.T)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context_vector = np.dot(attention_weights, V)
    return attention_weights, context_vector
```

Question2)


## **📘 Overview**
This file implements a **simplified Transformer Encoder Block** using **PyTorch**, following the architecture introduced in the paper  
**"Attention Is All You Need" (Vaswani et al., 2017)**.

The encoder block contains:
- Multi-Head Self-Attention  
- Feed-Forward Network (FFN)  
- Residual Connections (Add)  
- Layer Normalization (Norm)  

All components are implemented using PyTorch modules.

---

## **🧩 Components Implemented**

### **1. Multi-Head Self-Attention**
Uses:
```python
nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
````

### **2. Feed-Forward Network**

A two-layer network with ReLU:

```python
Linear(128 → 512 → 128)
```

### **3. Add & Norm**

Residual connections + LayerNorm are applied after:

* Multi-head attention
* Feed-forward network

---

## **⚙️ Model Dimensions**

* `d_model = 128`
* `num_heads = 8`
* Feed-forward expands to `4 * d_model = 512`
* Input shape tested: **[batch=32, seq_len=10, d_model=128]**

---

## **🚀 How to Run**

### **1. Install PyTorch**

```bash
pip install torch
```

### **2. Run the encoder block test**

Inside the project folder:

```bash
python3 transformer_encoder.py
```

### **Expected Output**

```
Output shape: torch.Size([32, 10, 128])
```

This confirms that:

* The encoder block works correctly
* Residual & normalization layers preserve shape
* Multi-head attention and FFN are functioning

---

## **📁 File Included**
