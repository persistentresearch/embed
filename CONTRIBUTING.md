# Contributing to Persistent Embed

Thank you for wanting to contribute. Persistent Embed is a community project - every contribution matters.

---

## Ways to Contribute

### Compute Contributions
The single most impactful contribution right now. We need GPU/TPU time to train models.

If you have spare compute — a GPU server running overnight, university cluster access, cloud credits — please reach out:
- Email: research@persistentresearch.in
- Subject: "Compute contribution - Persistent Embed"
- Include: hardware specs, available hours per week, duration

### Data Contributions
High-quality text pairs in Indian languages for training and evaluation.

What we need:
- Query-passage pairs in Hindi, Telugu, Tamil, Kannada, Bengali, Marathi
- Parallel sentence pairs across language combinations
- Domain-specific corpora (legal, medical, academic, news)

Open an issue titled "Data Contribution - [Language] - [Domain]" and describe what you have.

### Code Contributions

#### Getting Started

```bash
# Fork the repository on GitHub
# Then clone your fork:
git clone https://github.com/YOUR_USERNAME/embed
cd embed

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Run tests
pytest tests/

# Check formatting
black persistent_embed/
isort persistent_embed/
flake8 persistent_embed/

# Commit
git add .
git commit -m "feat: describe your change clearly"

# Push and open a Pull Request
git push origin feature/your-feature-name
```

#### Good First Issues

Look for issues labelled `good first issue` on GitHub. These are scoped, well-defined tasks suitable for first-time contributors.

Current good first issues typically include:
- Adding evaluation on a new language
- Writing documentation for a module
- Adding a new benchmark dataset
- Improving error messages

### Financial Contributions

Persistent Embed runs on donated compute and community funding. Every rupee and dollar goes directly to compute costs.

- [GitHub Sponsors](https://github.com/sponsors/persistent-research)
- [Open Collective](https://opencollective.com/persistent-embed) - full financial transparency

Sponsors are acknowledged in the README and model cards.

---

## Code Standards

- **Formatting:** Black (line length 88)
- **Imports:** isort
- **Linting:** flake8
- **Type hints:** Required for all public functions
- **Docstrings:** Google style
- **Tests:** pytest, required for all new features
- **Commits:** Conventional commits format (`feat:`, `fix:`, `docs:`, `test:`)

---

## Pull Request Guidelines

1. One feature or fix per PR — keep it focused
2. All tests must pass before review
3. Update documentation if behaviour changes
4. Add tests for new functionality
5. Reference any related issues in the PR description

We aim to review PRs within 72 hours. For research contributions (new training methods, evaluation tasks), please open a Discussion first before implementing.

---

## Community Standards

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

We are building something for the Indian and global NLP community. Be kind, be patient, be constructive.

---

## Questions?

Open an issue, start a Discussion, or email research@persistentresearch.in.
