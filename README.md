# Data Driven Learning and Control for Dynamical Systems
This repository contains code blocks that can be used to solve Assignments for Data-Driven Learning course offered at Clemson University

## Course Website

This repository is set up as a Jekyll course website that can be deployed with GitHub Pages.

### Preview locally

Install Bundler if needed:

```bash
gem install bundler
```

Install Ruby dependencies:

```bash
bundle install
```

Run the site:

```bash
bundle exec jekyll serve
```

Open:

```text
http://localhost:4000
```

### Edit course metadata

Update `_config.yml` with the semester, instructor, email, GitHub Pages `url`, and `baseurl`.

### Deploy with GitHub Pages

1. Push this repository to GitHub.
2. Open the repository settings.
3. Go to `Pages`.
4. Set source to `GitHub Actions`.
5. Add a Jekyll deployment workflow under `.github/workflows/`.
