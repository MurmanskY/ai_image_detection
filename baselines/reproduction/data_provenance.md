# Data Provenance 复现说明

- 论文: "Data Provenance for Image Auto-Regressive Generation", ICLR 2026
- 官方代码: third_party/DataProvenanceIAR (vendored); 上游 URL: UNKNOWN
- 运行方式: 官方脚本, 对 LlamaGen / VAR 各 1000 样本
- Score: 与源 codebook 的距离/重构几何, 越大越偏离
- 输出: `outputs/dataprov_llamagen_original_1000.csv`, `outputs/dataprov_var_original_1000.csv`
- 状态: vendored; 已运行采样评估
- 与论文差异: 每模型 1000 样本采样, 非全量
- 未确认: 上游官方仓库 URL 与 commit 未核验
