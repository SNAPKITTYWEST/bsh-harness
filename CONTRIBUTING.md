# Contributing to Binary Substrate Harness (BSH)

## License Agreement

By contributing to this project, you agree that your contributions will be licensed under the GNU Affero General Public License v3 (AGPLv3).

## Code Style

- C: Follow Linux kernel style (K&R with modern extensions)
- AGPL-3.0 header on all files
- No external dependencies beyond Linux kernel ABI

## Testing

- All code must compile with `-Wall -Wextra`
- Reproducible builds verified with `sha256sum`
- Determinism verified via comparison of identical runs

## Reporting Issues

- Use GitHub Issues
- Include: compiler version, OS, reproduction steps
- Provide SHA-256 hash of binary for reproducibility tracking

## Pull Request Process

1. Create feature branch from `main`
2. Add AGPL headers to new files
3. Verify reproducible build
4. Create PR with clear description
5. Ensure CI passes

## Code Ownership

Copyright (C) 2026 SNAPKITTYWEST  
Licensed under GNU Affero General Public License v3
