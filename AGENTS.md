Do not run GPU-dependent code.
Do not launch training or full evaluation.
Only run static checks, compilation, and small CPU-only tests.
If a test requires CUDA, skip it and report that it was not run.