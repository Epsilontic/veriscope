# Veriscope

**Early-warning system for representation drift in machine learning models.**

Veriscope is an open research project developing tools to detect early signs of collapse in internal model diversity before they lead to brittle and unsafe behavior.  
Standard metrics often look healthy while representation structure quietly degrades.  
Veriscope provides **auditable, reproducible monitoring signals** to surface these hidden risks.

---

## Features

- 📉 **Representation drift detection**: monitors collapse in internal model diversity  
- 🧪 **Reproducible experiments**: validated on CIFAR-10 with ≤5% run-level false positives  
- 🔒 **Audit-ready logging**: tamper-evident traces for transparency and accountability  
- ⚡ **Lightweight overhead**: step-time slowdown in the 2–8% range

---

## Getting Started

⚠️ *This is an early-stage prototype.*  
The codebase is under active development; interfaces and outputs may change.  

Planned releases:  
- ✅ CIFAR-10 reproduction suite (PyTorch, ~5k LOC)  
- 🔜 Extension to language model fine-tuning runs  
- 🔜 Public Python package with verifier + benchmark tools  

---

## Documentation

- [Preprint: Finite Realism](https://doi.org/10.5281/zenodo.17226486)  
- Technical notes and experiment logs will be added as the project matures.

---

## Contributing

At this stage, contributions are welcome in the form of feedback, replication attempts, and methodological critiques.  
Please open an issue or contact the maintainer directly.  

---

## License

This project is dual-licensed:  

- **GNU Affero General Public License v3.0 (AGPL-3.0-only)**  
- **Commercial license** (available for organizations who prefer not to comply with AGPL terms)  

See the [LICENSE](./LICENSE) file for full details.  

Use under AGPL-3.0 is free.  
For commercial licensing inquiries, contact the maintainer.

---

## Contact

**Maintainer:** Craig Holmander  
📧 craig.holm@protonmail.com  
🌐 [ORCID Profile](https://orcid.org/0009-0002-3145-8498)  
