# Contributing

Thanks for taking the time. Contributions of any size are welcome, including
typo fixes and questions that turn into documentation.

## Before you start

For anything more than a small fix, open an issue first and describe what you
want to change. It saves you from building something that turns out not to fit.

## Setting up

```bash
git clone https://github.com/Osyna/OpenInstaLink.git
cd OpenInstaLink
cargo build --release --locked
./target/release/link-camera --version
```

Rust 1.92 or newer. The first build downloads Cargo dependencies and the ONNX Runtime build dependency. Optional local vision models are separate; see the README. They are not required to build or to run the camera controls.

## Running the tests

```bash
cargo test --locked
```


## Making a change

1. Branch off `main`.
2. Keep the change focused: one concern per pull request.
3. Match the style already in the file you are editing.
4. Update the README or docs if you changed behaviour anyone depends on.
5. Write a commit message that explains *why*, not just what.

## Pull requests

Describe what changed and how you verified it. Screenshots help for anything
visual. CI has to pass before review.

## Reporting bugs

Include what you did, what happened, what you expected, and the versions of
link-camera, your OS and your runtime. A minimal reproduction is worth more than a
long description.

## Code of conduct

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
