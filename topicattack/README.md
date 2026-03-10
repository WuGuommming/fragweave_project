# TopicAttack on BIPIA EmailQA

这个目录用于把 **TopicAttack: An Indirect Prompt Injection Attack via Topic Transition** 接到你当前的 FragWeave/BIPIA EmailQA 评测流里。

## 设计原则

- **不改动原有 FragWeave 主流程**，避免影响你自己的方法与已有结果。
- **复用现有的** EmailQA loader、target model、judge、detector、sanitizer，保证 ASR / localization / after-sanitizer 可直接公平对比。
- 默认实现的是更贴近原论文仓库的 **`original` variant**：
  1. 先从恶意 instruction 推出一个 benign-looking `topic`
  2. 再用 topic-transition prompt 生成多轮伪对话
  3. 按原仓库风格拼成：`[assistant][response] OK.` + conversation + final `[user][instruction] ... [data]`

## 运行

```bash
bash topicattack/scripts/run_emailqa_topicattack.sh
```

或者：

```bash
python topicattack/run_emailqa_topicattack.py \
  --config topicattack/configs/emailqa_topicattack.yaml
```

## 输出

默认写到：

- `topicattack/outputs/.../topicattack_emailqa_results.csv`
- `topicattack/outputs/.../topicattack_emailqa_summary.json`
- `topicattack/outputs/.../topicattack_emailqa_debug_examples.json`

## 关键说明

- localization GT 采用 **整段 TopicAttack block** 的 shadow-tag 方式，因此 transition 文本也计入 GT；这和你之前讨论的口径一致。
- direct baseline 也会同时跑，并在同一份结果里给出，方便横向比较。
- 目前只实现 **EmailQA**，但代码结构已经独立在 `topicattack/` 下，后续可以扩到别的任务。
