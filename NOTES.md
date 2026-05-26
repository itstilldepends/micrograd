# 学习笔记

> 个人学习记录。Claude 只在我明确说"记一下/改 NOTES"时才动这个文件。

---

## 进度

### 视频进度
- [ ] 0:00 看了导数的数值定义
- [ ] 25min Value 类、加法乘法
- [ ] 55min 手动反向传播
- [ ] 1:15 闭包自动反传
- [ ] 1:35 拓扑排序
- [ ] 1:45 tanh / 拆解
- [ ] 2:00 nn.py
- [ ] 2:15 训练 demo

### 代码掌握度
- [ ] 能不看代码默写出 `Value.__add__` 的 forward + `_backward`
- [ ] 能解释拓扑排序为什么要 DFS + 后序
- [ ] 能给 `Value` 加一个新 op(比如 `tanh` / `exp` / `sigmoid`),并自己验证梯度
- [ ] 能跑通 `demo.ipynb` 并解释每一步在做什么
- [ ] 能解释 `test_engine.py` 是怎么用 PyTorch 当 oracle 的

---

## 核心概念笔记

### autograd / 反向传播
<!-- 比如:为什么 grad 要 +=、拓扑排序为什么、backward 的 self.grad=1 含义 -->

### Python 机制
<!-- dunder methods、closure、lambda、生成器、运算符重载等,记自己版本的理解 -->

### nn 库(Neuron / Layer / MLP)
<!-- 网络结构、parameters() 扁平化、最后一层为什么不加非线性等 -->

### 训练 / demo.ipynb
<!-- loss 设计、SGD 步骤、batch 处理等 -->

---

## 问答精华

<!-- 自己觉得有价值的对话:问题 + 关键答案。一条一段,留时间戳。
格式示例:

### Q: dunder/闭包/lambda 是什么 (2026-05-25)
- dunder = 双下划线方法,Python 把 `a+b` 翻译成 `a.__add__(b)`
- lambda = 一行匿名函数,micrograd 里只用作 `_backward` 的默认占位
- closure = 嵌套函数捕获外层变量的"引用",这是 `_backward` 能在 `__add__` 返回后还读到正确 `out.grad` 的根因
-->

---

## TODO / 还没搞懂的

<!-- 留这里的问题,下次继续问 -->
