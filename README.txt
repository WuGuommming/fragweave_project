直接替换文件版。

把压缩包内的文件按相对路径覆盖到仓库根目录：
- fragweave/run_sweep.py
- fragweave/run_sweep_promptlocate.py
- fragweave/attacks/guidance.py
- fragweave/attacks/weaver.py

这版改动只修最小必要断链：
1. run_sweep.py 的 placement 识别新 slot
2. run_sweep.py / run_sweep_promptlocate.py 不再把 guide slot 写死成 guide
3. guidance.py 不再把 B/C 因 slot 冲突塌回 bridge
4. weaver.py 让 guide 也按 slot 语义融合
