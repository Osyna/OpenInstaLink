fn main() -> std::process::ExitCode {
    match link_camera::cli::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("link-camera: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
