use anyhow::Result;

mod compile_shaders;
mod install_warp;
mod vendor;

const HELP: &str = r#"xtask

USAGE:
    xtask [OPTIONS] <SUBCOMMAND>

OPTIONS:
    -h, --help    Print help information

SUBCOMMANDS:
    compile-shaders    Compile shaders for the project
    install-warp       Install WARP software renderer (Windows only)
    vendor             Download and extract FidelityFX SDK
"#;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let mut pargs = pico_args::Arguments::from_env();

    let subcommand = pargs.subcommand()?.unwrap_or_default();

    if subcommand.is_empty() {
        if pargs.contains(["-h", "--help"]) {
            print!("{HELP}");
            return Ok(());
        }
        println!("{HELP}");
        return Ok(());
    }

    match subcommand.as_str() {
        "compile-shaders" => compile_shaders::compile_shaders(pargs)?,
        "install-warp" => install_warp::run_install_warp(pargs)?,
        "vendor" => vendor::vendor(pargs)?,
        cmd => {
            eprintln!("Unknown subcommand: {cmd}");
            eprintln!();
            println!("{HELP}");
            return Err(anyhow::anyhow!("Unknown subcommand: {cmd}"));
        }
    }

    Ok(())
}
