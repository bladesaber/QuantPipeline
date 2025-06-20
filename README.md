# QuantPipeline

* FrameWork  
  * Data
  * Math-Models
  * Simulation
    * model
    * non-model
      * Back-Test
  * Strategy
    * Cross-Section Factor(Method)
    * Time-Series Factor(Method)
    * Agent
      * Factor Signal Rules
      * Risk Signal Rules
    * Optimization Method
      * gradient
      * non-gradient
  * Portfolio
    * Optimization
      * Quadratic Programming
  * Analysis

# Reference Package:
  - pytimetk:https://github.com/business-science/pytimetk | easy for analysis and visulization
  - merlion:https://github.com/salesforce/Merlion,machine | Automatic machine learning for time series
  - darts:https://github.com/unit8co/darts                | Automatic machine learning for time series
  - Nixtla:https://github.com/Nixtla                      | Automatic classicial/machine learning model for ecosystem time series
  - pyod:https://github.com/yzhao062/pyod                 | Outlier and Anomaly Detection
  - pgmpy:https://github.com/pgmpy/pgmpy                  | Python library for Probabilistic Graphical Models
  - pomegranate:https://github.com/jmschrei/pomegranate   | Pytorch of PGM
  - daft:https://github.com/daft-dev/daft                 | Probabilistic Graphical Model (PGM) visulization
  - pyro:https://github.com/pyro-ppl/pyro                 | Pytorch for Probabilistic programming languages (PPLs)
  - pymc:https://github.com/pymc-devs/pymc                | Probabilistic programming for bayesian statistical model focusing on advanced Markov chain Monte Carlo (MCMC) and variational inference (VI)
  - TorchRL:https://github.com/pytorch/rl                 | reinforcement learning
  - acme:https://github.com/google-deepmind/acme          | reinforcement learning
  - RLlib:https://docs.ray.io/en/latest/rllib/index.html  | multi reinforcement learning
  - FinRl:https://github.com/AI4Finance-Foundation/FinRL  | reinforcement learning for finance
  - FinGPT:https://github.com/AI4Finance-Foundation/FinGPT| GPT for finance
  - hugging face:https://huggingface.co/docs              | LLM
  - OpenRLHF:https://github.com/OpenRLHF/OpenRLHF
  - tianshou:https://github.com/thu-ml/tianshou           | reinforcement learning
  - dash:https://github.com/plotly/dash                   | interation and visulization


## Ideas
- 2024/12/25
  - 先完成对架构的划分
  - 统计方法产出的只能是【相对定位因子(横截面)】或【时序因子】
  - 统计目的要不是relative pricing就是front running
  - Signal按由简单->完备演变: Direction -> Direction,Strength -> Distribution
  - Signal必须依赖锚定物,锚定物要不是未来时刻的自身,就是同时刻其他标的
  - Signal System要不是portfolio weight就是个series decision system
  - 我只是写了个大概理念，但可能代码是错误的，需要验证与测试
  - 如果特征能被稀疏表述，例如字典学习/PCA/clustring之类，那就能根据分块计算structure break的可能性，作为label
  - 如果特征能被稀疏表述，那time series同样能被表述为有限马尔科夫过程，为预测提供更多思路
