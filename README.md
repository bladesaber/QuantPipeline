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


## Ideas
- 2024/12/25
  - 先完成对架构的划分
  - 统计方法产出的只能是【相对定位因子(横截面)】或【时序因子】
  - 统计目的要不是relative pricing就是front running
  - Signal按由简单->完备演变: Direction -> Direction,Strength -> Distribution
  - Signal必须依赖锚定物,锚定物要不是未来时刻的自身,就是同时刻其他标的
  - Signal System要不是portfolio weight就是个series decision system
