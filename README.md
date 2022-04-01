# bergwald.github.io

Bergwald's Github pages site.

The static website is built with Jekyll and uses a customized version of the [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) theme.

Customizations to the theme include:

- ASCII art logo 
- Smaller font-sizes
- Cupertino as the syntax highlighting theme
- Fira Code as the monospace font
- Mathjax to render equations written in LaTeX
- Notices have been co-opted to serve as containers for code cell outputs
- Disabled post pagination
- Disabled social share links
- The header of the table of contents has been removed
- Larger and symmetric margins for vertical elements
- A custom style for bibliographies

For local development, run `bundle exec jekyll serve`. By default, the Jekyll dev server only responds to requests to the address `127.0.0.1`. To access the dev server from the local network, direct Jekyll to listen to all IPv4 addresses on the local machine by specifying the `host` flag: `bundle exec jekyll serve --host 0.0.0.0`.
