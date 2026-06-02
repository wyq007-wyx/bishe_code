# 论文 LaTeX 框架说明

主文件：`tebs_paper_framework.tex`

建议使用 XeLaTeX 编译，因为正文包含中文且使用了 `ctex`：

```powershell
xelatex tebs_paper_framework.tex
xelatex tebs_paper_framework.tex
```

当前文件参考 `论文模版` 下的 IEEE 双栏论文结构，并结合以下材料组织内容：

- `完整实验方案.md`
- `docs/文章背景.pdf`
- `docs/问题建模.pdf`
- `docs/求解方案设计.pdf`
- `docs/对比方案设计.pdf`
- `docs/实验任务选型与任务集构建.pdf`
- `docs/默认参数设置依据.md`

注意：文中标注“占位数值”“示例”“待真实实验替换”的表格和曲线只用于论文框架与排版效果展示，正式论文应使用真实实验输出替换。
