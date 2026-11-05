fn main() {
    #[cfg(feature = "gui")]
    {
        let config = slint_build::CompilerConfiguration::new().with_style("fluent-light".into());
        slint_build::compile_with_config("ui/app.slint", config).expect("compile Slint camera UI");
    }
}
