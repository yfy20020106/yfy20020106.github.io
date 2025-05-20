---
title: Adversarial Training
published: 2025-05-20
description: 'Adversarial Training Learning Notes'
image: ''
tags: [machine-learing, deep-learing]
category: 'Learning-Notes'
draft: false 
lang: ''
---
参考博客：https://kexue.fm/archives/7234

## Min-Max

对抗训练可以统一写成如下格式
$$
\min_{\theta}\mathbb{E}_{(x,y)\sim \mathcal{D}} \big[\max_{\Delta x \in \Omega }L(x+\Delta x,y;\theta)\big]
$$
其中$\mathcal{D}$代表训练集，$x$代表输入，$y$代表标签，$\theta$表示模型参数，$L$是损失函数（针对单个样本的loss），$\Delta x$是对抗扰动，$\Omega$是扰动空间。

这个式子可以分步理解：

1. 向$x$注入扰动$\Delta x$，$\Delta x$的目标是$\max\{L(x+\Delta x,y;\theta)\}$。
2. $\Delta x$通常会有约束，满足$\|\Delta x\|\le \epsilon$，其中$\epsilon$是一个常数。
3. 对每个样本$x$构造出对抗样本$x+\Delta x$后后，用$(x+\Delta x, y)$作为数据去最小化loss（更新$\theta$），这一步就是上述式子最外层的$\min$。
4. 反复交替执行1、2、3步。

## 快速梯度法（FGN）

为了注入扰动$\Delta x$，我们首先得设计方法来获得$\Delta x$确保对抗样本的loss一定是大于原先样本的loss。这个方法很好设计，只需要梯度上升即可，我们取
$$
\Delta x = \epsilon \nabla_xL(x,y;\theta)
$$
为了防止$\Delta x$过大，通常要对$\nabla_xL(x,y;\theta)$做一些标准化操作，比较常见的方式是
$$
\Delta x =\epsilon \frac{\nabla_xL(x,y;\theta)}{ \|\nabla_xL(x,y;\theta)\|} \quad or \quad \Delta x = \epsilon \text{sign}(\nabla_xL(x,y;\theta))
$$

## 投影梯度法（PGD）

通过迭代$T$次来获得最终对抗样本。
$$
x_{t+1} = \prod_{x+\mathcal{S}}(x_t+\alpha\frac{g(x_t)}{\|g(x_t)\|_2})\\
g(x_t) = \nabla_x L(\theta,x_t, y)
$$
其中$\mathcal{S} = r \in \mathbb{R}^d:\|r\|_2\le\epsilon$为扰动的约束空间，$\alpha$步长。

### 投影操作

- $L_{\infty}$约束：逐元素裁剪到$[x-\epsilon, x+\epsilon]$

- $L_2$约束：若扰动范数超过$\epsilon$，则归一化：
  $$
  \text{Proj}_{x+\mathcal{S}}(x') = x + \frac{(x'-x)\cdot \epsilon}{\max(\|x'-x\|_2,\epsilon)}
  $$
  
  