# Migration invariants (Phase A/B)  
  
These are non-negotiable invariants while refactoring the launch surface.  
  
## Invariants  
  
1) CIFAR training semantics must not change  
- The legacy runner remains the source of truth.  
- The new CLI may only change how it is invoked, not what it does.  
- Default output directory semantics remain the same as legacy.  
  
2) Lazy imports  
- Importing `veriscope.cli.main` must not import `torchvision` unless the legacy runner is invoked.  
  
3) Resolved config artifact  
- Every `veriscope run gpt ...` and `veriscope run cifar ...` writes:  
  - `<outdir>/run_config_resolved.json`  
- Must include:  
  - argv (wrapper + delegated)  
  - relevant env (SCAR_*, VERISCOPE_*, CUDA_* essentials)  
  - package version  
  - git SHA when available  
  - timestamp + run_id  
  
4) Smoke commands are reproducible  
- Scripts must:  
  - set minimal env explicitly  
  - create a unique outdir  
  - print the resolved command  
  - exit nonzero on failure  
