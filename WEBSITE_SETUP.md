# Website Setup Notes

These notes are for maintaining the course website. Keep student-facing course instructions in `README.md`.

## Local Environment

Use a dedicated conda environment for the Jekyll tooling:

```bash
conda create -n course-site -c conda-forge ruby=3.2 c-compiler cxx-compiler make cmake pkg-config
conda activate course-site
gem install bundler --bindir "$CONDA_PREFIX/bin"
```

Always activate the conda environment before running Bundler or Jekyll:

```bash
conda activate course-site
```

If native gem compilation fails with errors like `x86_64-conda-linux-gnu-cc: not found`, the conda environment is not fully activated in that shell.

## Install Dependencies

From the repository root:

```bash
bundle install
```

If the lockfile was created from a broken or stale environment, remove it and regenerate:

```bash
rm Gemfile.lock
bundle install
```

## Preview Locally

Run:

```bash
bundle exec jekyll serve
```

Open:

```text
http://127.0.0.1:4000/
```

The template currently emits Sass deprecation warnings. They come from the upstream template style files and do not prevent the site from building.

## Template Structure

This site follows the structure of:

```text
https://kazemnejad.github.io/jekyll-course-website-template/
```

Important files:

- `_config.yml`: course name, semester, school, URL, and build settings
- `_sass/_user_vars.scss`: Clemson color theme
- `index.md`: homepage content, using `layout: home`
- `_data/nav.yml`: top navigation
- `_data/people.yml`: instructors and teaching assistants
- `_announcements/`: manual homepage announcements
- `_lectures/`: lecture entries
- `_assignments/`: assignment entries
- `_events/`: schedule events and deadlines
- `_layouts/`: page templates copied/adapted from the course template
- `_includes/`: reusable template fragments
- `_css/main.scss` and `_sass/`: template styling
- `_images/`: logo, header pattern, and profile photos

## Editing Course Metadata

Update `_config.yml`:

```yaml
course_name: "Data Driven Learning and Control for Dynamical Systems"
course_description: "..."
course_semester: "Fall 20XX"
schoolname: "Clemson University"
schoolurl: "https://www.clemson.edu"
school_logo: "/_images/clemson-logo.jpg"
lab_logo: "/_images/lab-logo.png"
baseurl: ""
url: ""
```

## Updating Logos And Theme

The header uses two configurable logos from `_config.yml`:

```yaml
school_logo: "/_images/clemson-logo.jpg"
lab_logo: "/_images/lab-logo.png"
```

Replace these files with the real images:

- `_images/clemson-logo.jpg`
- `_images/lab-logo.png`

The Clemson color theme is in `_sass/_user_vars.scss`:

```scss
$primary-color: #522D80;
$accent-color: #F56600;
$title-color: #522D80;
$header-text-color: #fff;
$header-color: #522D80;
```

For GitHub Pages project sites, set:

```yaml
url: "https://YOUR-GITHUB-USERNAME.github.io"
baseurl: "/REPOSITORY-NAME"
```

For organization or user sites, `baseurl` is usually empty.

## Adding People

Edit `_data/people.yml`:

```yaml
instructors:
  - name: "Instructor Name"
    profile_pic: /_images/pp/instructor.jpg
    webpage: "https://example.com"

teaching_assistants:
  - name: "TA Name"
    profile_pic: /_images/pp/ta-1.jpg
    webpage: ""
```

Put profile images in `_images/pp/`.

## Adding Announcements

Create a file in `_announcements/`:

```markdown
---
date: 2026-08-20T09:00:00-04:00
---

Welcome to the course website.
```

Announcements appear on the homepage under `Updates`.

## Adding Lectures

Create a file in `_lectures/`:

```markdown
---
title: "Lecture Title"
type: lecture
date: 2026-08-25T09:00:00-04:00
tldr: "Short lecture summary."
links:
  - url: /static_files/presentations/lecture-01.pdf
    name: slides
  - url: /RESOURCES/edmd/
    name: codes
---

Additional lecture notes can go here.
```

Lecture entries appear in both `Lectures` and homepage updates unless `hide_from_announcments: true` is set.

## Adding Assignments

Create a file in `_assignments/`:

```markdown
---
title: "Assignment 1"
type: assignment
date: 2026-09-01T09:00:00-04:00
pdf: /static_files/assignments/assignment-1.pdf
attachment: /static_files/assignments/assignment-1.zip
due_event:
  type: due
  date: 2026-09-15T23:59:00-04:00
  description: "Assignment 1 due"
---

Assignment instructions can go here.
```

Assignment entries appear in `Assignments`, `Schedule`, and homepage updates unless `hide_from_announcments: true` is set.

## Adding Schedule Events

Create a file in `_events/`:

```markdown
---
type: raw_event
date: 2026-09-10T09:00:00-04:00
description: "No class"
---

Optional event details.
```

Supported event types copied from the template include:

- `raw_event`
- `due`
- `exam`

## GitHub Pages Deployment

For a project repository:

1. Push the repository to GitHub.
2. Set `url` and `baseurl` in `_config.yml`.
3. In GitHub, open repository `Settings`.
4. Go to `Pages`.
5. Use GitHub Actions or the repository branch deployment flow.

If using GitHub Actions, add a Jekyll workflow under `.github/workflows/`.
