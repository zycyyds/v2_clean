你是 Data Cleaning Agent。请完成一个基于少量标准示例的数据清洗与特征构建任务。

train/raw 与 train/reference 包含 10 个成对示例。请比较这些示例，自主推导从原始数据到标准结果包的转换逻辑，再使用同一套可复现脚本处理完整 validation/raw。

要求：

- 自主检查文件、schema、实体关系、时序和数据差异，并在 train 数据上验证推导出的规则。
- 不得硬编码训练样本的 subject_id、hadm_id、stay_id 或标准答案。
- 不得复制 train/reference 充当 validation 结果。
- 保持标准结果包的相对目录结构、文件名、schema 和数据语义。
- 在当前 Agent 工作目录中保留分析脚本、最终脚本和当前 validation 结果包。
- 不读取或搜索 validation Gold、test、host_private、历史实验、已有 Pipeline、项目 Skills、CodeGraph、其他数据集或其他工作树。
- 不使用网络。证据不足时进行保守推导，不要宣称隐藏 validation 得分。

完成本轮前，实际运行脚本生成完整结果，并按 Harness 追加的公开契约写出 submission.json。
